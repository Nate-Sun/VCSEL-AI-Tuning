from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from ipaddress import IPv4Address
from math import isfinite
import socket
from typing import Protocol


DEFAULT_BERT_HOST = "192.168.3.2"
DEFAULT_BERT_PORT = 9000
DEFAULT_TIMEOUT_S = 2.0
RECEIVE_BUFFER_SIZE = 4096
FIR_TAP_COUNT = 7


class BertCommunicationError(RuntimeError):
    """Raised when the UDP request cannot be completed."""


class BertProtocolError(RuntimeError):
    """Raised when the BERT returns an invalid or untrusted response."""


class DatagramSocket(Protocol):
    def settimeout(self, timeout: float) -> None: ...

    def bind(self, address: tuple[str, int]) -> None: ...

    def sendto(self, data: bytes, address: tuple[str, int]) -> int: ...

    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]: ...

    def close(self) -> None: ...


SocketFactory = Callable[[int, int], DatagramSocket]


@dataclass(frozen=True)
class BertConfiguration:
    """Parsed BERT channel state in the project's canonical tap order."""

    port: str
    channel: str
    taps: tuple[int, int, int, int, int, int, int]
    upper_nlc: int
    lower_nlc: int
    raw_response: str
    pattern_fields: tuple[str, str] = ("", "")

    @property
    def vendor_taps(self) -> tuple[int, int, int, int, int, int, int]:
        """Return taps as Pre1, Pre2, Pre3, Main, Post1, Post2, Post3."""
        pre3, pre2, pre1, main, post1, post2, post3 = self.taps
        return pre1, pre2, pre3, main, post1, post2, post3


def _single_ascii_field(value: str | int, label: str) -> str:
    text = str(value)
    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must be a single ASCII field") from error
    if not text or any(character.isspace() for character in text):
        raise ValueError(f"{label} must be a single ASCII field")
    return text


def _parse_integer_fields(
    fields: list[str], indexes: range | tuple[int, ...], label: str
) -> tuple[int, ...]:
    try:
        return tuple(int(fields[index], 10) for index in indexes)
    except ValueError as error:
        raise BertProtocolError(
            f"BERT {label} fields must be base-10 integers"
        ) from error


def parse_configuration(
    raw_response: str,
    *,
    port: str,
    channel: str | int,
) -> BertConfiguration:
    """Parse ``GetConfiguration`` using the commissioned VIT field layout."""
    port_text = _single_ascii_field(port, "port")
    channel_text = _single_ascii_field(channel, "channel")
    fields = raw_response.split()
    if len(fields) < 15:
        raise BertProtocolError(
            "BERT configuration response must contain at least 15 fields, "
            f"got {len(fields)}"
        )

    vendor_taps = _parse_integer_fields(fields, range(7), "FIR taps")
    upper_nlc, lower_nlc = _parse_integer_fields(fields, (13, 14), "NLC")
    pre1, pre2, pre3, main, post1, post2, post3 = vendor_taps
    internal_taps = (pre3, pre2, pre1, main, post1, post2, post3)
    return BertConfiguration(
        port=port_text,
        channel=channel_text,
        taps=internal_taps,
        upper_nlc=upper_nlc,
        lower_nlc=lower_nlc,
        raw_response=raw_response,
        pattern_fields=(fields[10], fields[12]),
    )


class VitBertClient:
    """Strict single-request UDP client for the VIT8X00 BERT API."""

    def __init__(
        self,
        host: str = DEFAULT_BERT_HOST,
        port: int = DEFAULT_BERT_PORT,
        *,
        local_ip: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        socket_factory: SocketFactory | None = None,
    ) -> None:
        try:
            self.host = str(IPv4Address(host))
        except ValueError as error:
            raise ValueError("host must be an IPv4 address") from error
        if type(port) is not int or not 1 <= port <= 65_535:
            raise ValueError("port must be an integer from 1 through 65535")
        self.port = port
        if local_ip is None:
            self.local_ip = None
        else:
            try:
                self.local_ip = str(IPv4Address(local_ip))
            except ValueError as error:
                raise ValueError("local_ip must be an IPv4 address") from error
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not isfinite(float(timeout_s))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive finite number")
        self.timeout_s = float(timeout_s)
        self._socket_factory = socket_factory or socket.socket

    @property
    def endpoint(self) -> tuple[str, int]:
        return self.host, self.port

    def _request(self, command: str) -> str:
        try:
            payload = command.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("BERT command must contain only ASCII") from error

        udp_socket: DatagramSocket | None = None
        try:
            udp_socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.settimeout(self.timeout_s)
            if self.local_ip is not None:
                udp_socket.bind((self.local_ip, 0))
            udp_socket.sendto(payload, self.endpoint)
            response, sender = udp_socket.recvfrom(RECEIVE_BUFFER_SIZE)
        except OSError as error:
            raise BertCommunicationError(
                f"BERT UDP request {command!r} failed: {error}"
            ) from error
        finally:
            if udp_socket is not None:
                udp_socket.close()

        if sender != self.endpoint:
            raise BertProtocolError(
                f"unexpected UDP sender {sender!r}; expected {self.endpoint!r}"
            )
        try:
            decoded = response.decode("ascii")
        except UnicodeDecodeError as error:
            raise BertProtocolError("BERT response must be ASCII") from error
        if not decoded.strip():
            raise BertProtocolError("BERT response must not be empty")
        return decoded

    def get_mode(self) -> str:
        """Return the mode reported by ``GetMode``."""
        return self._request("GetMode").strip()

    def get_configuration(
        self, port: str, channel: str | int
    ) -> BertConfiguration:
        """Read and parse one port/channel configuration."""
        port_text = _single_ascii_field(port, "port")
        channel_text = _single_ascii_field(channel, "channel")
        raw_response = self._request(
            f"GetConfiguration {port_text} {channel_text}"
        )
        return parse_configuration(
            raw_response,
            port=port_text,
            channel=channel_text,
        )

    def set_fir(
        self,
        port: str,
        channel: str | int,
        *,
        taps: Sequence[int],
        upper_nlc: int,
        lower_nlc: int,
    ) -> str:
        """Send one ``setFir`` command after converting canonical tap order."""
        port_text = _single_ascii_field(port, "port")
        channel_text = _single_ascii_field(channel, "channel")
        try:
            internal_taps = tuple(taps)
        except TypeError as error:
            raise TypeError("taps must contain exactly seven integers") from error
        if len(internal_taps) != FIR_TAP_COUNT or any(
            type(tap) is not int for tap in internal_taps
        ):
            raise ValueError("taps must contain exactly seven integers")
        if type(upper_nlc) is not int or type(lower_nlc) is not int:
            raise TypeError("NLC values must be integers")

        pre3, pre2, pre1, main, post1, post2, post3 = internal_taps
        values = (
            pre1,
            pre2,
            pre3,
            main,
            post1,
            post2,
            post3,
            upper_nlc,
            lower_nlc,
        )
        command = (
            f"setFir {port_text} {channel_text} "
            + " ".join(str(value) for value in values)
        )
        return self._request(command)
