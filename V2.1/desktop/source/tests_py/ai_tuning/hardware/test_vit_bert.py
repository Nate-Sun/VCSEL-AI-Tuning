from __future__ import annotations

import socket

import pytest

from vcsel_ai_tuning.hardware.vit_bert import (
    BertCommunicationError,
    BertProtocolError,
    VitBertClient,
    parse_configuration,
)


LIVE_CONFIGURATION = (
    "-26 5 0 110 -6 -6 -1 off on off SSPRQ N/A SSPRQ "
    "6 6 0 ns enabled txDisabled rxDisabled"
)


class FakeDatagramSocket:
    def __init__(
        self,
        response: bytes,
        sender: tuple[str, int] = ("192.168.3.2", 9000),
        receive_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.sender = sender
        self.receive_error = receive_error
        self.timeout: float | None = None
        self.bound_to: tuple[str, int] | None = None
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.receive_sizes: list[int] = []
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def bind(self, address: tuple[str, int]) -> None:
        self.bound_to = address

    def sendto(self, data: bytes, address: tuple[str, int]) -> int:
        self.sent.append((data, address))
        return len(data)

    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]:
        self.receive_sizes.append(size)
        if self.receive_error is not None:
            raise self.receive_error
        return self.response, self.sender

    def close(self) -> None:
        self.closed = True


class SocketFactory:
    def __init__(self, fake_socket: FakeDatagramSocket) -> None:
        self.fake_socket = fake_socket
        self.calls: list[tuple[int, int]] = []

    def __call__(self, family: int, kind: int) -> FakeDatagramSocket:
        self.calls.append((family, kind))
        return self.fake_socket


def _client(
    fake_socket: FakeDatagramSocket,
    *,
    local_ip: str | None = "192.168.3.1",
    timeout_s: float = 1.25,
) -> tuple[VitBertClient, SocketFactory]:
    factory = SocketFactory(fake_socket)
    client = VitBertClient(
        host="192.168.3.2",
        port=9000,
        local_ip=local_ip,
        timeout_s=timeout_s,
        socket_factory=factory,
    )
    return client, factory


def test_parse_live_configuration_maps_vendor_taps_to_internal_order():
    configuration = parse_configuration(
        LIVE_CONFIGURATION,
        port="port1",
        channel="0",
    )

    assert configuration.port == "port1"
    assert configuration.channel == "0"
    assert configuration.taps == (0, 5, -26, 110, -6, -6, -1)
    assert configuration.vendor_taps == (-26, 5, 0, 110, -6, -6, -1)
    assert configuration.upper_nlc == 6
    assert configuration.lower_nlc == 6
    assert configuration.pattern_fields == ("SSPRQ", "SSPRQ")
    assert configuration.raw_response == LIVE_CONFIGURATION


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("-26 5 0 110 -6 -6", "at least 15"),
        (LIVE_CONFIGURATION.replace("-26", "bad", 1), "FIR taps"),
        (LIVE_CONFIGURATION.replace("6 6 0 ns", "bad 6 0 ns"), "NLC"),
    ],
)
def test_parse_configuration_rejects_malformed_responses(response, reason):
    with pytest.raises(BertProtocolError, match=reason):
        parse_configuration(response, port="port1", channel="0")


def test_get_mode_uses_default_endpoint_optional_bind_timeout_and_ascii_once():
    fake_socket = FakeDatagramSocket(b"PAM4 53.125 1lanePerPort")
    client, factory = _client(fake_socket)

    mode = client.get_mode()

    assert mode == "PAM4 53.125 1lanePerPort"
    assert factory.calls == [(socket.AF_INET, socket.SOCK_DGRAM)]
    assert fake_socket.timeout == pytest.approx(1.25)
    assert fake_socket.bound_to == ("192.168.3.1", 0)
    assert fake_socket.sent == [(b"GetMode", ("192.168.3.2", 9000))]
    assert fake_socket.receive_sizes == [4096]
    assert fake_socket.closed


def test_get_mode_skips_bind_when_local_ip_is_not_configured():
    fake_socket = FakeDatagramSocket(b"PAM4")
    client, _ = _client(fake_socket, local_ip=None)

    assert client.get_mode() == "PAM4"
    assert fake_socket.bound_to is None


def test_get_configuration_sends_exact_command_and_parses_response():
    fake_socket = FakeDatagramSocket(LIVE_CONFIGURATION.encode("ascii"))
    client, _ = _client(fake_socket)

    configuration = client.get_configuration("port1", "0")

    assert fake_socket.sent == [
        (b"GetConfiguration port1 0", ("192.168.3.2", 9000))
    ]
    assert configuration.taps == (0, 5, -26, 110, -6, -6, -1)
    assert configuration.upper_nlc == 6
    assert configuration.lower_nlc == 6


def test_set_fir_reorders_internal_taps_and_sends_exactly_once():
    fake_socket = FakeDatagramSocket(b"OK")
    client, _ = _client(fake_socket)

    response = client.set_fir(
        "port1",
        "0",
        taps=(0, 5, -26, 110, -6, -6, -1),
        upper_nlc=6,
        lower_nlc=6,
    )

    assert response == "OK"
    assert fake_socket.sent == [
        (
            b"setFir port1 0 -26 5 0 110 -6 -6 -1 6 6",
            ("192.168.3.2", 9000),
        )
    ]
    assert fake_socket.receive_sizes == [4096]


def test_rejects_reply_from_any_endpoint_other_than_configured_device():
    fake_socket = FakeDatagramSocket(
        b"PAM4",
        sender=("192.168.3.99", 9000),
    )
    client, _ = _client(fake_socket)

    with pytest.raises(BertProtocolError, match="unexpected UDP sender"):
        client.get_mode()

    assert len(fake_socket.sent) == 1
    assert fake_socket.closed


def test_rejects_non_ascii_device_response():
    fake_socket = FakeDatagramSocket(b"PAM4\xff")
    client, _ = _client(fake_socket)

    with pytest.raises(BertProtocolError, match="ASCII"):
        client.get_mode()


def test_wraps_udp_timeout_and_always_closes_socket():
    fake_socket = FakeDatagramSocket(
        b"",
        receive_error=socket.timeout("timed out"),
    )
    client, _ = _client(fake_socket)

    with pytest.raises(BertCommunicationError, match="timed out"):
        client.get_mode()

    assert len(fake_socket.sent) == 1
    assert fake_socket.closed


@pytest.mark.parametrize(
    "bad_taps",
    [
        (0, 5, -26, 110, -6, -6),
        (0, 5, -26, 110, -6, -6, 1.5),
        (0, 5, -26, 110, -6, -6, True),
    ],
)
def test_set_fir_rejects_invalid_internal_taps_before_opening_socket(bad_taps):
    fake_socket = FakeDatagramSocket(b"OK")
    client, factory = _client(fake_socket)

    with pytest.raises((TypeError, ValueError), match="seven integers"):
        client.set_fir(
            "port1",
            "0",
            taps=bad_taps,
            upper_nlc=6,
            lower_nlc=6,
        )

    assert factory.calls == []


@pytest.mark.parametrize("port,channel", [("port 1", "0"), ("port1", "0\nMute")])
def test_rejects_command_field_injection_before_opening_socket(port, channel):
    fake_socket = FakeDatagramSocket(b"OK")
    client, factory = _client(fake_socket)

    with pytest.raises(ValueError, match="single ASCII field"):
        client.get_configuration(port, channel)

    assert factory.calls == []
