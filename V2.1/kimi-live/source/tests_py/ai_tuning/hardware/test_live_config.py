from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from vcsel_ai_tuning.hardware.live_config import (
    BertConfig,
    CharacterizationConfig,
    DcaConfig,
    ErMeasurementConfig,
    I2cConfig,
    LiveHardwareConfig,
    ParameterFieldConfig,
    TdecqMeasurementConfig,
    load_live_hardware_config,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_CONFIG = REPO_ROOT / "configs" / "hardware" / "vcsel_lab_live.yaml"


def _valid_document() -> dict[str, object]:
    return {
        "dca": {
            "resource": "TCPIP0::Nate-Rog::hislip0,4880::INSTR",
            "expected_idn_prefix": "Keysight Technologies,N1010A,",
            "function": 1,
            "timeout_ms": 60000,
        },
        "bert": {
            "ip": "192.168.3.2",
            "local_ip": "192.168.3.1",
            "udp_port": 9000,
            "port": "port1",
            "channel": "0",
            "timeout_s": 2,
        },
        "i2c": {
            "port": "COM7",
            "baudrate": 115200,
            "device_address": 0xA6,
            "chip_id_register": 0x0004,
            "expected_chip_id": 0x0B,
            "chip_revision_register": 0x0005,
            "expected_chip_revision": 0x0D,
        },
        "characterization": {
            "default_channel": 0,
            "baseline_repeats": 5,
            "measurement_timeout_s": 180,
            "poll_interval_s": 1,
            "meaningful_er_floor_db": None,
            "meaningful_tdecq_floor_db": None,
            "tdecq": {"source": "FUNCtion1"},
            "er": {
                "source": "CHAN1A",
                "value_query": ":MEASure:EYE:OER?",
                "source_query": ":MEASure:EYE:OER:SOURce?",
                "status_query": ":MEASure:EYE:OER:STATus?",
                "details_query": ":MEASure:EYE:OER:STATus:DETails?",
                "units_query": ":MEASure:EYE:OER:UNITs?",
            },
            "parameters": {
                "ibias_dac": {"address": 0x2002, "bit_start": 0, "bits": 7},
                "efin_cur": {"address": 0x2003, "bit_start": 0, "bits": 4},
                "vga_cur": {"address": 0x2005, "bit_start": 0, "bits": 4},
                "eq": {"address": 0x2006, "bit_start": 0, "bits": 5},
                "res_gain": {"address": 0x2007, "bit_start": 0, "bits": 4},
                "res_bw": {"address": 0x2008, "bit_start": 0, "bits": 4},
                "gain_ctrl": {"address": 0x2009, "bit_start": 0, "bits": 6},
                "vbias_ref": {"address": 0x200A, "bit_start": 0, "bits": 4},
                "vref": {"address": 0x200B, "bit_start": 0, "bits": 4},
            },
            "safety_limits": {},
        },
    }


def _write_yaml(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_repository_live_config_loads_exact_verified_lab_endpoints():
    config = load_live_hardware_config(LIVE_CONFIG)

    assert config == LiveHardwareConfig(
        dca=DcaConfig(
            resource="TCPIP0::Nate-Rog::hislip0,4880::INSTR",
            expected_idn_prefix="Keysight Technologies,N1010A,",
            function=1,
            timeout_ms=60000,
        ),
        bert=BertConfig(
            ip="192.168.3.2",
            local_ip="192.168.3.1",
            udp_port=9000,
            port="port1",
            channel="0",
            timeout_s=2.0,
        ),
        i2c=I2cConfig(
            port="COM7",
            baudrate=115200,
            device_address=0xA6,
            chip_id_register=0x0004,
            expected_chip_id=0x52,
            chip_revision_register=0x0005,
            expected_chip_revision=0x00,
        ),
        characterization=CharacterizationConfig(
            default_channel=0,
            baseline_repeats=5,
            measurement_timeout_s=180.0,
            poll_interval_s=1.0,
            meaningful_er_floor_db=None,
            meaningful_tdecq_floor_db=None,
            tdecq=TdecqMeasurementConfig(source="FUNC1"),
            er=ErMeasurementConfig(
                source="CHAN1A",
                value_query=":MEASure:EYE:OER?",
                source_query=":MEASure:EYE:OER:SOURce?",
                status_query=":MEASure:EYE:OER:STATus?",
                details_query=":MEASure:EYE:OER:STATus:DETails?",
                units_query=":MEASure:EYE:OER:UNITs?",
            ),
            parameters=(
                ParameterFieldConfig("ibias_dac", 0x2002, 0, 7),
                ParameterFieldConfig("efin_cur", 0x2003, 0, 4),
                ParameterFieldConfig("vga_cur", 0x2005, 0, 4),
                ParameterFieldConfig("eq", 0x2006, 0, 5),
                ParameterFieldConfig("res_gain", 0x2007, 0, 4),
                ParameterFieldConfig("res_bw", 0x2008, 0, 4),
                ParameterFieldConfig("gain_ctrl", 0x2009, 0, 6),
                ParameterFieldConfig("vbias_ref", 0x200A, 0, 4),
                ParameterFieldConfig("vref", 0x200B, 0, 4),
            ),
        ),
    )
    with pytest.raises(FrozenInstanceError):
        config.bert.ip = "192.0.2.1"


def test_integer_bert_channel_is_normalized_to_string(tmp_path):
    document = _valid_document()
    document["bert"]["channel"] = 7

    config = load_live_hardware_config(_write_yaml(tmp_path, document))

    assert config.bert.channel == "7"


@pytest.mark.parametrize("section", ["dca", "bert", "i2c"])
def test_missing_required_section_is_rejected(tmp_path, section):
    document = _valid_document()
    del document[section]

    with pytest.raises(ValueError, match=rf"required.*{section}|{section}.*required"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("dca", "resource"),
        ("bert", "local_ip"),
        ("i2c", "expected_chip_revision"),
    ],
)
def test_missing_required_field_is_rejected(tmp_path, section, field):
    document = _valid_document()
    del document[section][field]

    with pytest.raises(ValueError, match=rf"{section}\.{field}.*required"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("dca", "resource", 4880),
        ("dca", "function", True),
        ("bert", "ip", 19216832),
        ("bert", "channel", True),
        ("i2c", "port", 7),
        ("i2c", "baudrate", 115200.5),
    ],
)
def test_wrong_field_type_is_rejected(tmp_path, section, field, value):
    document = _valid_document()
    document[section][field] = value

    with pytest.raises(ValueError, match=rf"{section}\.{field}.*type"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("dca", "resource", ""),
        ("dca", "function", 0),
        ("dca", "timeout_ms", -1),
        ("bert", "ip", "999.1.1.1"),
        ("bert", "udp_port", 65536),
        ("bert", "port", "  "),
        ("bert", "channel", ""),
        ("bert", "timeout_s", 0),
        ("i2c", "port", ""),
        ("i2c", "baudrate", 0),
        ("i2c", "device_address", 0x100),
        ("i2c", "chip_id_register", -1),
        ("i2c", "expected_chip_id", 0x100),
        ("i2c", "chip_revision_register", 0x10000),
        ("i2c", "expected_chip_revision", -1),
    ],
)
def test_out_of_range_field_is_rejected(tmp_path, section, field, value):
    document = _valid_document()
    document[section][field] = value

    with pytest.raises(ValueError, match=rf"{section}\.{field}"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


@pytest.mark.parametrize("document", [None, [], "not a mapping"])
def test_non_mapping_yaml_root_is_rejected(tmp_path, document):
    with pytest.raises(ValueError, match="root.*mapping"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


@pytest.mark.parametrize(
    "source",
    [
        "CHAN1A",
        "CHAN9A",
        "CHAN1E",
        "FUNC0",
        "FUNC2",
        "FUNC17",
        "not-a-waveform",
    ],
)
def test_tdecq_measurement_source_requires_func1(tmp_path, source):
    document = _valid_document()
    document["characterization"]["tdecq"]["source"] = source

    with pytest.raises(ValueError, match="characterization.tdecq.source"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


def test_tdecq_source_is_per_measurement_and_independent_from_er(tmp_path):
    document = _valid_document()
    document["characterization"]["tdecq"]["source"] = "FUNCtion1"
    document["characterization"]["er"]["source"] = "CHANnel1A"

    config = load_live_hardware_config(_write_yaml(tmp_path, document))
    assert not hasattr(config.dca, "measurement_source")
    assert config.characterization.tdecq.source == "FUNC1"
    assert config.characterization.er.source == "CHAN1A"


def test_outer_er_measurement_source_requires_chan1a(tmp_path):
    document = _valid_document()
    document["characterization"]["er"]["source"] = "FUNC1"

    with pytest.raises(ValueError, match="characterization.er.source.*CHAN1A"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


def test_missing_tdecq_measurement_source_is_rejected(tmp_path):
    document = _valid_document()
    del document["characterization"]["tdecq"]["source"]

    with pytest.raises(ValueError, match="characterization.tdecq.source.*required"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


def test_characterization_parameters_support_channels_zero_through_three(tmp_path):
    config = load_live_hardware_config(_write_yaml(tmp_path, _valid_document()))
    parameters = {item.name: item for item in config.characterization.parameters}

    assert list(parameters) == [
        "ibias_dac",
        "efin_cur",
        "vga_cur",
        "eq",
        "res_gain",
        "res_bw",
        "gain_ctrl",
        "vbias_ref",
        "vref",
    ]
    assert parameters["ibias_dac"].address_for_channel(0) == 0x2002
    assert parameters["ibias_dac"].address_for_channel(1) == 0x2102
    assert parameters["vref"].address_for_channel(3) == 0x230B
    with pytest.raises(ValueError, match="channel"):
        parameters["eq"].address_for_channel(4)
    assert all(item.safety_min is None for item in parameters.values())


def test_characterization_rejects_partial_safety_limits(tmp_path):
    document = _valid_document()
    document["characterization"]["safety_limits"] = {
        "eq": {"minimum": 10, "maximum": 25}
    }

    with pytest.raises(ValueError, match="safety.*eq.*max_step|eq.*max_step"):
        load_live_hardware_config(_write_yaml(tmp_path, document))


def test_characterization_applies_complete_operator_safety_limits(tmp_path):
    document = _valid_document()
    document["characterization"]["safety_limits"] = {
        "eq": {"minimum": 18, "maximum": 25, "max_step": 2}
    }

    config = load_live_hardware_config(_write_yaml(tmp_path, document))
    eq = next(item for item in config.characterization.parameters if item.name == "eq")

    assert (eq.safety_min, eq.safety_max, eq.max_step) == (18, 25, 2)
