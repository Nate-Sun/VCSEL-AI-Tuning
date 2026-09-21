from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.v1_1.dataset import decode_png_dimensions
from vcsel_ai_tuning.v1_1.live_dataset_capture import FlexDcaDatasetCapture


def acquisition_guard():
    return {
        "commands": {
            ":MEASure:EYE:TDEQ:SOURce?": "FUNC1",
            ":MEASure:EYE:TDEQ:STATus?": "CORR",
            ":MEASure:EYE:TDEQ:COUNt?": "17",
            ":MEASure:EYE:TDEQ?": "2.41",
            ":MEASure:EYE:OER:SOURce?": "CHAN1A",
            ":MEASure:EYE:OER:STATus?": "CORR",
            ":MEASure:EYE:OER:COUNt?": "17",
            ":MEASure:EYE:OER?": "5.72",
        },
        "errors": {},
        "original_source": "FUNC1",
        "original_source_raw": "FUNC1",
        "source_restore": {
            "readback": "FUNC1",
            "verified": True,
        },
        "primary_match": True,
    }


class FakeBinarySession:
    def __init__(
        self,
        *,
        invalid_source=None,
        refuse_restore=False,
        waveform_source="WMEM1",
        measurement_source="FUNC1",
        refused_measurement_writes=(),
        failed_measurement_writes=(),
        refused_measurement_write_occurrences=None,
        query_failures=None,
        response_overrides=None,
    ):
        self.source = waveform_source
        self.measurement_source = measurement_source
        self.invalid_source = invalid_source
        self.refuse_restore = refuse_restore
        self.refused_measurement_writes = set(refused_measurement_writes)
        self.failed_measurement_writes = set(failed_measurement_writes)
        self.refused_measurement_write_occurrences = {
            key: set(occurrences)
            for key, occurrences in dict(
                refused_measurement_write_occurrences or {}
            ).items()
        }
        self.measurement_write_counts = {}
        self.query_failures = dict(query_failures or {})
        self.response_overrides = dict(response_overrides or {})
        self.writes = []
        self.queries = []
        self.binary_calls = []
        self.density = {
            "CHAN1A": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32),
            "FUNC1": np.array([[7, 8, 9], [10, 11, 12]], dtype=np.uint32),
        }
        self.measurements = {
            ":MEASure:EYE:OER": {
                "CHAN1A": {"status": "CORR", "count": "17", "value": "5.72"},
                "FUNC1": {"status": "CORR", "count": "17", "value": "8.01"},
            },
            ":MEASure:EYE:TDEQ": {
                "CHAN1A": {"status": "CORR", "count": "17", "value": "6.15"},
                "FUNC1": {"status": "CORR", "count": "17", "value": "2.41"},
            },
            ":MEASure:EYE:PAM:LINearity": {
                "CHAN1A": {"status": "CORR", "value": "0.97"},
                "CHAN2A": {"status": "CORR", "value": "0.87"},
                "FUNC1": {"status": "CORR", "value": "0.51"},
            },
            ":MEASure:EYE:OOMA": {
                "CHAN1A": {"status": "CORR", "units": "WATT", "value": "0.0012"},
                "FUNC1": {"status": "CORR", "units": "WATT", "value": "0.0091"},
            },
            ":MEASure:EYE:APOWer": {
                "CHAN1A": {"status": "CORR", "units": "WATT", "value": "0.0024"},
                "FUNC1": {"status": "CORR", "units": "WATT", "value": "0.0092"},
            },
        }

    def write(self, command):
        self.writes.append(command)
        if command.startswith(":WAVeform:SOURce "):
            requested = command.split()[-1]
            if not (self.refuse_restore and requested == "WMEM1"):
                self.source = requested
            return
        if ":SOURce " in command and command.startswith(":MEASure:EYE:"):
            stem, requested = command.split(":SOURce ", 1)
            stem = stem.rstrip(":")
            key = (stem, requested)
            occurrence = self.measurement_write_counts.get(key, 0) + 1
            self.measurement_write_counts[key] = occurrence
            if key in self.failed_measurement_writes:
                raise RuntimeError(f"injected write failure for {command}")
            refused_occurrences = self.refused_measurement_write_occurrences.get(
                key, set()
            )
            if (
                key not in self.refused_measurement_writes
                and occurrence not in refused_occurrences
            ):
                self.measurement_source = requested

    def query(self, command):
        self.queries.append((command, self.measurement_source, self.source))
        remaining_failures = self.query_failures.get(command, 0)
        if remaining_failures:
            self.query_failures[command] = remaining_failures - 1
            raise RuntimeError(f"injected query failure for {command}")
        if command == ":WAVeform:SOURce?":
            return self.source
        if command == ":SYSTem:BORDer?":
            return "LENDian"
        if command == ":WAVeform:EYE:STATus?":
            return "INV" if self.source == self.invalid_source else "CORR"
        if command == ":WAVeform:EYE:STATus:REASon?":
            return "injected invalid waveform"
        if command == ":WAVeform:EYE:ROWS?":
            return "2"
        if command == ":WAVeform:EYE:COLumns?":
            return "3"
        if command == ":WAVeform:EYE:POINts?":
            return "6"
        if command in {
            ":WAVeform:EYE:XINCrement?",
            ":WAVeform:EYE:XORigin?",
            ":WAVeform:EYE:YINCrement?",
            ":WAVeform:EYE:YORigin?",
        }:
            return "0.25"
        if command in self.response_overrides:
            return self.response_overrides[command]
        for stem, by_source in self.measurements.items():
            if command == f"{stem}:SOURce?":
                return self.measurement_source
            measurement = by_source.get(
                self.measurement_source,
                {"status": "INV", "count": "0", "value": "9.91E+37"},
            )
            if command == f"{stem}:STATus?":
                return measurement["status"]
            if command == f"{stem}:COUNt?":
                return measurement.get("count", "0")
            if command == f"{stem}:UNITs?":
                return measurement.get("units", "")
            if command == f"{stem}?":
                return measurement["value"]
        responses = {
            ":SPRocess1:TEQualizer:TAPS?": "-0.1,1.0,0.2",
            ":SPRocess1:TEQualizer:NPRecursors?": "1",
        }
        return responses[command]

    def query_binary_values(self, command, **kwargs):
        self.binary_calls.append((command, kwargs))
        return self.density[self.source].reshape(-1, order="F").tolist()


def test_flexdca_capture_decodes_same_acquisition_raw_and_equalized_eyes():
    session = FakeBinarySession()
    ownership_checks = []

    def exclusive_access():
        ownership_checks.append(
            (len(session.queries), len(session.binary_calls), session.source)
        )
        return True

    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=exclusive_access,
    )

    capture = adapter.capture(
        "acquisition-1",
        SimpleNamespace(
            raw_responses={
                "primary": "same acquisition",
                "acquisition_guard": acquisition_guard(),
            }
        ),
    )

    np.testing.assert_array_equal(capture.raw_eye_density, session.density["CHAN1A"])
    assert capture.density_axis_order == ("amplitude", "time")
    assert capture.density_source == "CHAN1A"
    assert decode_png_dimensions(capture.tdecq_equalized_eye_png) == (3, 2)
    assert capture.rlm == pytest.approx(0.97)
    assert capture.oma == pytest.approx(0.0012)
    assert capture.average_power == pytest.approx(0.0024)
    assert capture.equalizer_taps == (-0.1, 1.0, 0.2)
    assert capture.equalizer_main_index == 1
    assert capture.equalizer_source == "FUNC1"
    assert capture.same_acquisition_verified is True
    assert capture.raw_responses["same_acquisition"]["consistent"] is True
    assert capture.raw_responses["same_acquisition"] == {
        "method": "source_bound_frozen_snapshot",
        "isolation": "in_process_resource_scope",
        "cross_process_limitation": (
            "does_not_prevent_other_processes_or_hosts_from_accessing_instrument"
        ),
        "primary": acquisition_guard(),
        "before_dataset_queries": capture.raw_responses["same_acquisition"][
            "before_dataset_queries"
        ],
        "after_dataset_queries": capture.raw_responses["same_acquisition"][
            "after_dataset_queries"
        ],
        "consistent": True,
        "verified": True,
        "exclusive_access": {
            "before": True,
            "after": True,
            "errors": {},
        },
        "no_acquisition_command_issued": True,
    }
    assert session.source == "WMEM1"
    assert session.writes[0:3] == [
        ":MEASure:EYE:OER:SOURce CHAN1A",
        ":MEASure:EYE:TDEQ:SOURce FUNC1",
        ":MEASure:EYE:TDEQ:SOURce FUNC1",
    ]
    assert session.writes[-1] == ":WAVeform:SOURce WMEM1"
    assert len(session.binary_calls) == 2
    assert ownership_checks[0] == (0, 0, "WMEM1")
    assert ownership_checks[1] == (
        len(session.queries),
        len(session.binary_calls),
        "WMEM1",
    )
    assert all(call[1]["datatype"] == "L" for call in session.binary_calls)
    assert not any("ACQuire" in command for command in session.writes)


def test_flexdca_capture_restores_waveform_source_when_data_is_invalid():
    session = FakeBinarySession(invalid_source="FUNC1")
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    with pytest.raises(RuntimeError, match="FUNC1.*INV"):
        adapter.capture("acquisition-2", SimpleNamespace(raw_responses={}))

    assert session.source == "WMEM1"
    assert session.writes[-1] == ":WAVeform:SOURce WMEM1"


def test_flexdca_capture_marks_unprovable_acquisition_without_faking_success():
    session = FakeBinarySession()
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    capture = adapter.capture(
        "acquisition-3",
        SimpleNamespace(raw_responses={}),
    )

    assert capture.same_acquisition_verified is False
    assert capture.raw_responses["same_acquisition"]["verified"] is False


@pytest.mark.parametrize(
    "exclusive_access",
    (None, lambda: False),
    ids=("absent", "false"),
)
def test_consistent_capture_without_exclusive_ownership_remains_unverified(
    exclusive_access,
):
    session = FakeBinarySession()
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=exclusive_access,
    )

    capture = adapter.capture(
        "acquisition-no-exclusive-ownership",
        SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
    )

    same_acquisition = capture.raw_responses["same_acquisition"]
    assert same_acquisition["consistent"] is True
    assert same_acquisition["exclusive_access"] == {
        "before": False,
        "after": False,
        "errors": {},
    }
    assert same_acquisition["verified"] is False
    assert capture.same_acquisition_verified is False


def test_exclusive_ownership_lost_after_dataset_queries_remains_unverified():
    session = FakeBinarySession()
    ownership_checks = iter((True, False))
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: next(ownership_checks),
    )

    capture = adapter.capture(
        "acquisition-exclusive-ownership-lost",
        SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
    )

    same_acquisition = capture.raw_responses["same_acquisition"]
    assert same_acquisition["consistent"] is True
    assert same_acquisition["exclusive_access"] == {
        "before": True,
        "after": False,
        "errors": {},
    }
    assert same_acquisition["verified"] is False
    assert capture.same_acquisition_verified is False


def test_exclusive_verifier_exception_is_audited_and_fails_closed():
    session = FakeBinarySession()
    checks = iter((RuntimeError("injected verifier failure"), True))

    def exclusive_access():
        result = next(checks)
        if isinstance(result, Exception):
            raise result
        return result

    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=exclusive_access,
    )

    capture = adapter.capture(
        "acquisition-exclusive-verifier-error",
        SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
    )

    same_acquisition = capture.raw_responses["same_acquisition"]
    assert same_acquisition["exclusive_access"] == {
        "before": False,
        "after": True,
        "errors": {"before": "injected verifier failure"},
    }
    assert same_acquisition["verified"] is False
    assert capture.same_acquisition_verified is False


def test_flexdca_capture_rejects_noncallable_exclusive_verifier():
    with pytest.raises(TypeError, match="exclusive_access must be callable"):
        FlexDcaDatasetCapture(
            session=FakeBinarySession(),
            raw_source="CHAN1A",
            equalized_source="FUNC1",
            equalizer_function=1,
            exclusive_access=True,
        )


def test_primary_guard_is_detached_before_first_dataset_query():
    commands = dict(acquisition_guard()["commands"])
    commands[":MEASure:EYE:OER?"] = "99.0"
    guard_backing = acquisition_guard()
    guard_backing["commands"] = MappingProxyType(commands)
    primary_responses = MappingProxyType(
        {"acquisition_guard": MappingProxyType(guard_backing)}
    )

    class MutatingSession(FakeBinarySession):
        def query(self, command):
            if not self.queries:
                commands[":MEASure:EYE:OER?"] = "5.72"
            return super().query(command)

    session = MutatingSession()
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )

    capture = adapter.capture(
        "acquisition-detached-primary-guard",
        SimpleNamespace(raw_responses=primary_responses),
    )

    same_acquisition = capture.raw_responses["same_acquisition"]
    assert commands[":MEASure:EYE:OER?"] == "5.72"
    assert same_acquisition["primary"]["commands"][
        ":MEASure:EYE:OER?"
    ] == "99.0"
    assert same_acquisition["consistent"] is False
    assert same_acquisition["verified"] is False
    assert capture.same_acquisition_verified is False


def test_invalid_primary_evidence_shape_is_rejected_before_dca_queries():
    session = FakeBinarySession()
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )

    with pytest.raises(RuntimeError, match="primary measurement evidence"):
        adapter.capture(
            "acquisition-invalid-primary-evidence",
            SimpleNamespace(
                raw_responses={"acquisition_guard": {"commands": object()}}
            ),
        )

    assert session.queries == []
    assert session.writes == []


def test_nonmapping_primary_guard_is_rejected_before_dca_queries():
    session = FakeBinarySession()
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )

    with pytest.raises(RuntimeError, match="acquisition_guard must be a mapping"):
        adapter.capture(
            "acquisition-invalid-primary-guard",
            SimpleNamespace(raw_responses={"acquisition_guard": []}),
        )

    assert session.queries == []
    assert session.writes == []


def test_flexdca_capture_rejects_unverified_source_restore():
    session = FakeBinarySession(refuse_restore=True)
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    with pytest.raises(RuntimeError, match="restore.*WMEM1"):
        adapter.capture(
            "acquisition-4",
            SimpleNamespace(
                raw_responses={"acquisition_guard": acquisition_guard()}
            ),
        )


def test_waveform_restore_write_failure_carries_completed_evidence():
    class RestoreWriteFailureSession(FakeBinarySession):
        def write(self, command):
            if command == ":WAVeform:SOURce WMEM1":
                self.writes.append(command)
                raise RuntimeError("injected waveform restore write failure")
            return super().write(command)

    session = RestoreWriteFailureSession()
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )

    with pytest.raises(RuntimeError, match="waveform source restore failed") as caught:
        adapter.capture(
            "acquisition-waveform-restore-write-failure",
            SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
        )

    evidence = caught.value.capture_evidence
    assert evidence["source_restore"] == {
        "expected": "WMEM1",
        "original_source_raw": "WMEM1",
        "readback": "FUNC1",
        "verified": False,
        "write_error": "injected waveform restore write failure",
    }
    assert session.source == "FUNC1"


def test_waveform_restore_query_failure_carries_completed_evidence():
    class RestoreQueryFailureSession(FakeBinarySession):
        def __init__(self):
            super().__init__()
            self.waveform_source_queries = 0

        def query(self, command):
            if command == ":WAVeform:SOURce?":
                self.waveform_source_queries += 1
                if self.waveform_source_queries == 4:
                    self.queries.append((command, self.measurement_source, self.source))
                    raise RuntimeError("injected waveform restore query failure")
            return super().query(command)

    session = RestoreQueryFailureSession()
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )

    with pytest.raises(RuntimeError, match="waveform source restore failed") as caught:
        adapter.capture(
            "acquisition-waveform-restore-query-failure",
            SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
        )

    evidence = caught.value.capture_evidence
    assert evidence["source_restore"] == {
        "expected": "WMEM1",
        "original_source_raw": "WMEM1",
        "readback": None,
        "verified": False,
        "readback_error": "injected waveform restore query failure",
    }
    assert session.source == "WMEM1"


def test_optional_metrics_bind_raw_source_and_restore_shared_source():
    session = FakeBinarySession(measurement_source="FUNC1")
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    capture = adapter.capture(
        "acquisition-source-bound-metrics",
        SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
    )

    expected = {
        ":MEASure:EYE:PAM:LINearity": (None, "0.97"),
        ":MEASure:EYE:OOMA": ("WATT", "0.0012"),
        ":MEASure:EYE:APOWer": ("WATT", "0.0024"),
    }
    for stem, (units, value) in expected.items():
        evidence = capture.raw_responses[stem]
        assert evidence == {
            "original_source": "FUNC1",
            "original_source_raw": "FUNC1",
            "requested_source": "CHAN1A",
            "verified_source": "CHAN1A",
            "status": "CORR",
            "units": units,
            "value": value,
            "source_restore": {
                "expected": "FUNC1",
                "readback": "FUNC1",
                "verified": True,
            },
        }
        assert f"{stem}:SOURce CHAN1A" in session.writes
        assert f"{stem}:SOURce FUNC1" in session.writes
    assert capture.rlm == pytest.approx(0.97)
    assert capture.oma == pytest.approx(0.0012)
    assert capture.average_power == pytest.approx(0.0024)
    assert session.measurement_source == "FUNC1"
    assert not any("ACQuire" in command for command in session.writes)


def test_optional_metric_uses_nondefault_configured_raw_source():
    session = FakeBinarySession(measurement_source="FUNC1")
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="channel2a",
        equalized_source="FUNC1",
        equalizer_function=1,
    )
    raw = {}

    value = adapter._optional_metric(
        ":MEASure:EYE:PAM:LINearity",
        raw=raw,
    )

    assert value == pytest.approx(0.87)
    assert raw[":MEASure:EYE:PAM:LINearity"]["requested_source"] == "CHAN2A"
    assert raw[":MEASure:EYE:PAM:LINearity"]["verified_source"] == "CHAN2A"
    assert session.writes == [
        ":MEASure:EYE:PAM:LINearity:SOURce CHAN2A",
        ":MEASure:EYE:PAM:LINearity:SOURce FUNC1",
    ]
    assert session.measurement_source == "FUNC1"


def test_optional_metric_refused_restore_raises_instead_of_publishing():
    session = FakeBinarySession(
        refused_measurement_writes={(":MEASure:EYE:OOMA", "FUNC1")}
    )
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    with pytest.raises(RuntimeError, match="OOMA.*restore.*FUNC1"):
        adapter.capture(
            "acquisition-refused-metric-restore",
            SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
        )

    assert not any("ACQuire" in command for command in session.writes)


def test_acquisition_guard_binds_each_metric_and_restores_shared_source():
    session = FakeBinarySession(measurement_source="FUNC2")
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    snapshot = adapter._read_acquisition_guard(acquisition_guard())

    assert snapshot["commands"] == acquisition_guard()["commands"]
    assert snapshot["errors"] == {}
    assert snapshot["original_source"] == "FUNC2"
    assert snapshot["source_restore"] == {
        "expected": "FUNC2",
        "readback": "FUNC2",
        "verified": True,
    }
    assert session.measurement_source == "FUNC2"
    assert session.writes == [
        ":MEASure:EYE:OER:SOURce CHAN1A",
        ":MEASure:EYE:TDEQ:SOURce FUNC1",
        ":MEASure:EYE:TDEQ:SOURce FUNC2",
    ]
    assert (
        ":MEASure:EYE:OER:STATus?",
        "CHAN1A",
        "WMEM1",
    ) in session.queries
    assert (
        ":MEASure:EYE:TDEQ:STATus?",
        "FUNC1",
        "WMEM1",
    ) in session.queries
    assert not any("ACQuire" in command for command in session.writes)


@pytest.mark.parametrize(
    "response_overrides",
    [
        {":MEASure:EYE:OER:SOURce?": "CHAN2A"},
        {":MEASure:EYE:OER:COUNt?": "18"},
        {":MEASure:EYE:TDEQ?": "2.99"},
    ],
    ids=("source-drift", "count-drift", "value-drift"),
)
def test_acquisition_guard_drift_remains_inconsistent_and_unverified(
    response_overrides,
):
    session = FakeBinarySession(response_overrides=response_overrides)
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )

    capture = adapter.capture(
        "acquisition-drift",
        SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
    )

    same_acquisition = capture.raw_responses["same_acquisition"]
    assert same_acquisition["consistent"] is False
    assert same_acquisition["verified"] is False
    assert capture.same_acquisition_verified is False


def test_identical_nonfinite_guard_values_remain_unverified():
    guard = acquisition_guard()
    guard["commands"] = dict(guard["commands"])
    guard["commands"][":MEASure:EYE:TDEQ?"] = "9.91E+37"
    guard["commands"][":MEASure:EYE:OER?"] = "9.91E+37"
    session = FakeBinarySession(
        response_overrides={
            ":MEASure:EYE:TDEQ?": "9.91E+37",
            ":MEASure:EYE:OER?": "9.91E+37",
        }
    )
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )

    capture = adapter.capture(
        "acquisition-invalid-guard-values",
        SimpleNamespace(raw_responses={"acquisition_guard": guard}),
    )

    assert capture.raw_responses["same_acquisition"]["consistent"] is False
    assert capture.same_acquisition_verified is False


def test_guard_restore_source_drift_remains_unverified():
    session = FakeBinarySession(measurement_source="FUNC2")
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )

    capture = adapter.capture(
        "acquisition-guard-restore-source-drift",
        SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
    )

    same_acquisition = capture.raw_responses["same_acquisition"]
    assert same_acquisition["primary"]["original_source"] == "FUNC1"
    assert same_acquisition["before_dataset_queries"]["original_source"] == "FUNC2"
    assert same_acquisition["after_dataset_queries"]["original_source"] == "FUNC2"
    assert same_acquisition["consistent"] is False
    assert same_acquisition["verified"] is False
    assert capture.same_acquisition_verified is False


def test_primary_guard_sources_are_evidence_and_never_command_inputs():
    polluted_guard = acquisition_guard()
    polluted_guard["commands"] = dict(polluted_guard["commands"])
    polluted_guard["commands"][":MEASure:EYE:OER:SOURce?"] = "CHAN2A"
    polluted_guard["commands"][":MEASure:EYE:TDEQ:SOURce?"] = "FUNC2"
    session = FakeBinarySession()
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    capture = adapter.capture(
        "acquisition-polluted-primary-sources",
        SimpleNamespace(raw_responses={"acquisition_guard": polluted_guard}),
    )

    same_acquisition = capture.raw_responses["same_acquisition"]
    assert same_acquisition["consistent"] is False
    assert same_acquisition["verified"] is False
    assert capture.same_acquisition_verified is False
    assert same_acquisition["before_dataset_queries"]["commands"][
        ":MEASure:EYE:OER:SOURce?"
    ] == "CHAN1A"
    assert same_acquisition["before_dataset_queries"]["commands"][
        ":MEASure:EYE:TDEQ:SOURce?"
    ] == "FUNC1"
    measurement_source_writes = [
        command
        for command in session.writes
        if command.startswith(":MEASure:EYE:") and ":SOURce " in command
    ]
    assert measurement_source_writes
    assert all(
        command.endswith((" CHAN1A", " FUNC1"))
        for command in measurement_source_writes
    )
    assert not any(
        command.endswith((" CHAN2A", " FUNC2"))
        for command in measurement_source_writes
    )


def test_mid_query_guard_failure_still_restores_shared_source():
    session = FakeBinarySession(
        measurement_source="FUNC2",
        query_failures={":MEASure:EYE:OER:COUNt?": 1},
    )
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    snapshot = adapter._read_acquisition_guard(acquisition_guard())

    assert ":MEASure:EYE:OER:COUNt?" in snapshot["errors"]
    assert snapshot["source_restore"]["verified"] is True
    assert session.measurement_source == "FUNC2"
    assert session.writes[-1] == ":MEASure:EYE:TDEQ:SOURce FUNC2"


def test_guard_before_restore_write_failure_aborts_capture():
    restore_key = (":MEASure:EYE:TDEQ", "FUNC2")
    session = FakeBinarySession(
        measurement_source="FUNC2",
        failed_measurement_writes={restore_key},
    )
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    with pytest.raises(
        RuntimeError,
        match="acquisition guard.*source restore.*FUNC2",
    ) as caught:
        adapter.capture(
            "acquisition-guard-before-restore-failure",
            SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
        )

    evidence = caught.value.guard_snapshot
    assert evidence["source_restore"] == {
        "expected": "FUNC2",
        "readback": "FUNC1",
        "verified": False,
    }
    assert "source_restore" in evidence["errors"]
    assert session.source == "WMEM1"
    assert not any(
        command.startswith(":WAVeform:SOURce ") for command in session.writes
    )
    assert not any("ACQuire" in command for command in session.writes)


def test_guard_after_restore_mismatch_aborts_and_restores_waveform_source():
    restore_key = (":MEASure:EYE:TDEQ", "FUNC2")
    session = FakeBinarySession(
        measurement_source="FUNC2",
        refused_measurement_write_occurrences={restore_key: {2}},
    )
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    with pytest.raises(
        RuntimeError,
        match="acquisition guard.*source restore.*FUNC2",
    ) as caught:
        adapter.capture(
            "acquisition-guard-after-restore-mismatch",
            SimpleNamespace(raw_responses={"acquisition_guard": acquisition_guard()}),
        )

    evidence = caught.value.guard_snapshot
    assert evidence["source_restore"] == {
        "expected": "FUNC2",
        "readback": "FUNC1",
        "verified": False,
    }
    assert "source_restore" in evidence["errors"]
    assert session.source == "WMEM1"
    assert session.writes[-1] == ":WAVeform:SOURce WMEM1"
    assert not any("ACQuire" in command for command in session.writes)


def test_invalid_measurement_source_readback_is_audited_without_unsafe_write():
    injected = "FUNC1;:ACQuire:SINGle"
    session = FakeBinarySession(measurement_source=injected)
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )
    raw = {}

    with pytest.raises(RuntimeError, match="supported.*waveform"):
        adapter._optional_metric(":MEASure:EYE:OOMA", raw=raw, units=True)

    assert raw[":MEASure:EYE:OOMA"]["original_source_raw"] == injected
    assert session.writes == []


def test_invalid_waveform_source_readback_is_not_interpolated_for_restore():
    injected = "WMEM1;:ACQuire:SINGle"
    session = FakeBinarySession(waveform_source=injected)
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
    )

    with pytest.raises(RuntimeError, match="supported.*waveform"):
        adapter.capture("acquisition-invalid-restore", SimpleNamespace(raw_responses={}))

    assert session.writes == []
