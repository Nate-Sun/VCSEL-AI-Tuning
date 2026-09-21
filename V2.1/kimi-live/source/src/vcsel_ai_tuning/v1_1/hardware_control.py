"""Fail-closed V1.1 candidate transactions over existing hardware drivers.

This module contains no connection factory and performs no work at import or
construction time.  A caller must explicitly invoke an operation with an
already-created driver (LIVE) or a fake/replay implementation (tests).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration

from .regmap import MappingStatus, ParameterBinding, ParameterMapping


class HardwareTransactionError(RuntimeError):
    """A candidate failed, but the configured rollback outcome is known."""

    def __init__(self, message: str, *, rollback_verified: bool) -> None:
        self.rollback_verified = rollback_verified
        super().__init__(message)


class RecoveryRequired(HardwareTransactionError):
    """The single rollback attempt failed or could not be verified."""


class DcaMeasurementTimeout(RuntimeError):
    pass


class DcaConfigurationDrift(RuntimeError):
    def __init__(self, *, expected: str, current: str) -> None:
        self.expected = expected
        self.current = current
        super().__init__(
            f"DCA_CONFIGURATION_DRIFT: expected {expected}, read {current}"
        )


@dataclass(frozen=True, slots=True)
class BoundRegisterField:
    """VD1124 writer-compatible view of one verified RegMap binding."""

    name: str
    channel0_address: int
    channel_stride: int
    bit_start: int
    bits: int

    @classmethod
    def from_binding(cls, binding: ParameterBinding) -> "BoundRegisterField":
        if binding.status is not MappingStatus.AVAILABLE:
            raise ValueError(f"parameter {binding.name} is not available")
        assert binding.channel0_address is not None
        assert binding.channel_stride is not None
        assert binding.bit_start is not None
        assert binding.bits is not None
        return cls(
            name=binding.name,
            channel0_address=binding.channel0_address,
            channel_stride=binding.channel_stride,
            bit_start=binding.bit_start,
            bits=binding.bits,
        )

    @property
    def encoded_max(self) -> int:
        return (1 << self.bits) - 1

    def address_for_channel(self, channel: int) -> int:
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError("channel must be an integer")
        if not 0 <= channel <= 3:
            raise ValueError("channel must be from 0 through 3")
        return self.channel0_address + channel * self.channel_stride


class FieldValue(Protocol):
    name: str
    channel: int
    address: int
    raw_value: int
    value: int


class FieldWriteResult(Protocol):
    name: str
    channel: int
    address: int
    before_raw: int
    before_value: int
    requested_value: int
    after_raw: int
    after_value: int
    verified: bool


class FieldWriter(Protocol):
    def read_field(
        self, field: BoundRegisterField, *, channel: int
    ) -> FieldValue: ...

    def write_field(
        self,
        field: BoundRegisterField,
        *,
        channel: int,
        value: int,
        expected_current: int,
        expected_raw: int,
    ) -> FieldWriteResult: ...


@dataclass(frozen=True, slots=True)
class I2cWriteStep:
    name: str
    address: int
    before_value: int
    before_raw: int
    requested_value: int
    after_value: int
    after_raw: int

    def payload(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "address": self.address,
            "before_value": self.before_value,
            "before_raw": self.before_raw,
            "requested_value": self.requested_value,
            "after_value": self.after_value,
            "after_raw": self.after_raw,
        }


@dataclass(frozen=True, slots=True)
class I2cTransactionReceipt:
    requested: Mapping[str, int]
    readback: Mapping[str, int]
    steps: tuple[I2cWriteStep, ...]

    def payload(self) -> dict[str, object]:
        return {
            "requested": dict(self.requested),
            "readback": dict(self.readback),
            "steps": [step.payload() for step in self.steps],
        }


class I2cCandidateTransaction:
    """Apply one multi-field candidate with per-field RMW/readback guards."""

    def __init__(
        self,
        *,
        writer: FieldWriter,
        mapping: ParameterMapping,
        channel: int,
        b0_values: Mapping[str, int],
    ) -> None:
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError("channel must be an integer")
        if not 0 <= channel <= 3:
            raise ValueError("channel must be from 0 through 3")
        self._writer = writer
        self._mapping = mapping
        self._channel = channel
        self._b0_values = self._validate_values(b0_values, label="B0")

    def _ordered_bindings(
        self, values: Mapping[str, int]
    ) -> tuple[ParameterBinding, ...]:
        bindings = [self._mapping.require_available(name) for name in values]
        bindings.sort(
            key=lambda binding: (
                binding.script_order
                if binding.script_order is not None
                else 2**31
            )
        )
        return tuple(bindings)

    def _validate_values(
        self, values: Mapping[str, int], *, label: str
    ) -> dict[str, int]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{label} values must be a mapping")
        validated: dict[str, int] = {}
        for binding in self._ordered_bindings(values):
            value = values[binding.name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{binding.name} must be an integer")
            minimum, maximum = binding.encoding_range
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{binding.name} is outside hard-coded encoding range "
                    f"[{minimum}, {maximum}]"
                )
            validated[binding.name] = value
        return validated

    @staticmethod
    def _verify_read(
        binding: ParameterBinding,
        field: BoundRegisterField,
        value: FieldValue,
        channel: int,
    ) -> None:
        expected_address = field.address_for_channel(channel)
        if (
            value.name != binding.name
            or value.channel != channel
            or value.address != expected_address
            or isinstance(value.raw_value, bool)
            or not isinstance(value.raw_value, int)
            or isinstance(value.value, bool)
            or not isinstance(value.value, int)
            or value.value
            != ((value.raw_value >> field.bit_start) & field.encoded_max)
        ):
            raise RuntimeError(f"malformed I2C readback for {binding.name}")

    def _read(self, binding: ParameterBinding) -> tuple[BoundRegisterField, FieldValue]:
        field = BoundRegisterField.from_binding(binding)
        value = self._writer.read_field(field, channel=self._channel)
        self._verify_read(binding, field, value, self._channel)
        return field, value

    def _write_verified(
        self,
        binding: ParameterBinding,
        *,
        requested: int,
        before: FieldValue,
    ) -> I2cWriteStep:
        field = BoundRegisterField.from_binding(binding)
        result = self._writer.write_field(
            field,
            channel=self._channel,
            value=requested,
            expected_current=before.value,
            expected_raw=before.raw_value,
        )
        mask = field.encoded_max << field.bit_start
        expected_raw = (before.raw_value & ~mask) | (
            requested << field.bit_start
        )
        if (
            result.name != binding.name
            or result.channel != self._channel
            or result.address != field.address_for_channel(self._channel)
            or result.before_value != before.value
            or result.before_raw != before.raw_value
            or result.requested_value != requested
            or result.after_value != requested
            or result.after_raw != expected_raw
            or result.verified is not True
        ):
            raise RuntimeError(f"inconsistent I2C write receipt for {binding.name}")
        immediate = self._writer.read_field(field, channel=self._channel)
        self._verify_read(binding, field, immediate, self._channel)
        if immediate.value != requested or immediate.raw_value != expected_raw:
            raise RuntimeError(f"I2C immediate readback mismatch for {binding.name}")
        return I2cWriteStep(
            name=binding.name,
            address=result.address,
            before_value=result.before_value,
            before_raw=result.before_raw,
            requested_value=result.requested_value,
            after_value=result.after_value,
            after_raw=result.after_raw,
        )

    def _restore_b0_once(self) -> None:
        for binding in self._ordered_bindings(self._b0_values):
            field, current = self._read(binding)
            target = self._b0_values[binding.name]
            if current.value != target:
                self._write_verified(
                    binding, requested=target, before=current
                )
        for binding in self._ordered_bindings(self._b0_values):
            _, current = self._read(binding)
            if current.value != self._b0_values[binding.name]:
                raise RuntimeError(
                    f"B0 restore mismatch for {binding.name}: "
                    f"expected {self._b0_values[binding.name]}, read {current.value}"
                )

    def apply(self, requested: Mapping[str, int]) -> I2cTransactionReceipt:
        candidate = self._validate_values(requested, label="candidate")
        steps: list[I2cWriteStep] = []
        try:
            for binding in self._ordered_bindings(candidate):
                _, before = self._read(binding)
                target = candidate[binding.name]
                if before.value == target:
                    continue
                steps.append(
                    self._write_verified(
                        binding, requested=target, before=before
                    )
                )
            readback: dict[str, int] = {}
            for binding in self._ordered_bindings(candidate):
                _, current = self._read(binding)
                expected = candidate[binding.name]
                if current.value != expected:
                    raise RuntimeError(
                        f"I2C candidate readback mismatch for {binding.name}: "
                        f"expected {expected}, read {current.value}"
                    )
                readback[binding.name] = current.value
        except Exception as candidate_error:
            try:
                self._restore_b0_once()
            except Exception as rollback_error:
                raise RecoveryRequired(
                    "I2C candidate failed and B0 rollback could not be verified: "
                    f"candidate={candidate_error}; rollback={rollback_error}",
                    rollback_verified=False,
                ) from rollback_error
            raise HardwareTransactionError(
                f"I2C candidate failed; B0 rollback verified: {candidate_error}",
                rollback_verified=True,
            ) from candidate_error
        return I2cTransactionReceipt(
            requested=dict(candidate),
            readback=readback,
            steps=tuple(steps),
        )


class BertWriter(Protocol):
    def get_configuration(
        self, port: str, channel: str
    ) -> BertConfiguration: ...

    def set_fir(
        self,
        port: str,
        channel: str,
        *,
        taps: tuple[int, ...],
        upper_nlc: int,
        lower_nlc: int,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class BertTransactionReceipt:
    before_taps: tuple[int, ...]
    requested_taps: tuple[int, ...]
    readback_taps: tuple[int, ...]
    upper_nlc: int
    lower_nlc: int
    pattern: str
    port: str
    channel: str

    def payload(self) -> dict[str, object]:
        return {
            "before_taps": list(self.before_taps),
            "requested_taps": list(self.requested_taps),
            "readback_taps": list(self.readback_taps),
            "upper_nlc": self.upper_nlc,
            "lower_nlc": self.lower_nlc,
            "pattern": self.pattern,
            "port": self.port,
            "channel": self.channel,
        }


class BertCandidateTransaction:
    """Apply one complete seven-tap setFir and verify all frozen fields."""

    def __init__(
        self,
        *,
        client: BertWriter,
        port: str,
        channel: str,
        pattern: str,
        b0: BertConfiguration,
    ) -> None:
        if pattern != "SSPRQ":
            raise ValueError("BERT pattern must be fixed to SSPRQ")
        if b0.port != port or b0.channel != channel:
            raise ValueError("B0 BERT port/channel does not match transaction target")
        self._client = client
        self._port = port
        self._channel = channel
        self._pattern = pattern
        self._b0 = b0
        self._validate_configuration(b0, expected_taps=b0.taps)

    @staticmethod
    def _validate_taps(taps: tuple[int, ...]) -> tuple[int, ...]:
        try:
            result = tuple(taps)
        except TypeError as error:
            raise TypeError("FIR must contain seven integers") from error
        if len(result) != 7 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in result
        ):
            raise ValueError("FIR must contain seven integers")
        if result[3] <= 0:
            raise ValueError("FIR Main must be positive")
        l1 = sum(abs(value) for value in result)
        if l1 > 170:
            raise ValueError(f"FIR L1 {l1} exceeds 170")
        return result

    def _validate_configuration(
        self,
        configuration: BertConfiguration,
        *,
        expected_taps: tuple[int, ...],
    ) -> None:
        pattern_fields = getattr(configuration, "pattern_fields", ())
        if (
            configuration.port != self._port
            or configuration.channel != self._channel
            or configuration.taps != expected_taps
            or configuration.upper_nlc != self._b0.upper_nlc
            or configuration.lower_nlc != self._b0.lower_nlc
            or tuple(pattern_fields) != (self._pattern, self._pattern)
        ):
            raise RuntimeError(
                "BERT readback does not match requested taps and frozen "
                "port/channel/NLC/pattern"
            )

    def _restore_b0_once(self) -> None:
        self._client.set_fir(
            self._port,
            self._channel,
            taps=self._b0.taps,
            upper_nlc=self._b0.upper_nlc,
            lower_nlc=self._b0.lower_nlc,
        )
        restored = self._client.get_configuration(self._port, self._channel)
        self._validate_configuration(restored, expected_taps=self._b0.taps)

    def apply(self, taps: tuple[int, ...]) -> BertTransactionReceipt:
        requested = self._validate_taps(taps)
        before = self._client.get_configuration(self._port, self._channel)
        self._validate_configuration(before, expected_taps=before.taps)
        try:
            self._client.set_fir(
                self._port,
                self._channel,
                taps=requested,
                upper_nlc=self._b0.upper_nlc,
                lower_nlc=self._b0.lower_nlc,
            )
            readback = self._client.get_configuration(
                self._port, self._channel
            )
            self._validate_configuration(readback, expected_taps=requested)
        except Exception as candidate_error:
            try:
                self._restore_b0_once()
            except Exception as rollback_error:
                raise RecoveryRequired(
                    "BERT candidate failed and B0 rollback could not be verified: "
                    f"candidate={candidate_error}; rollback={rollback_error}",
                    rollback_verified=False,
                ) from rollback_error
            raise HardwareTransactionError(
                f"BERT candidate failed; B0 rollback verified: {candidate_error}",
                rollback_verified=True,
            ) from candidate_error
        return BertTransactionReceipt(
            before_taps=before.taps,
            requested_taps=requested,
            readback_taps=readback.taps,
            upper_nlc=readback.upper_nlc,
            lower_nlc=readback.lower_nlc,
            pattern=self._pattern,
            port=readback.port,
            channel=readback.channel,
        )


class DcaMeasurementAdapter(Protocol):
    def fingerprint(self) -> str: ...

    def measure_single(self) -> Any: ...


class DcaMeasurementController:
    """Enforce frozen DCA configuration and the one-reconnect timeout rule."""

    def __init__(
        self,
        *,
        adapter: DcaMeasurementAdapter,
        reconnect: Callable[[], DcaMeasurementAdapter],
        expected_fingerprint: str,
    ) -> None:
        if not expected_fingerprint:
            raise ValueError("expected_fingerprint must be nonblank")
        self._adapter = adapter
        self._reconnect = reconnect
        self._expected_fingerprint = expected_fingerprint

    def _verify_fingerprint(self, adapter: DcaMeasurementAdapter) -> None:
        current = adapter.fingerprint()
        if current != self._expected_fingerprint:
            raise DcaConfigurationDrift(
                expected=self._expected_fingerprint,
                current=current,
            )

    def measure_current_candidate(self) -> Any:
        self._verify_fingerprint(self._adapter)
        try:
            return self._adapter.measure_single()
        except DcaMeasurementTimeout:
            replacement = self._reconnect()
            self._verify_fingerprint(replacement)
            self._adapter = replacement
            return replacement.measure_single()
