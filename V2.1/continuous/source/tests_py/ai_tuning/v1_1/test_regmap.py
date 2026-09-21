import pytest

from vcsel_ai_tuning.v1_1.regmap import (
    SAFE_PARAMETER_NAMES,
    MappingStatus,
    ParameterSelection,
    ParameterSelectionError,
    RegMapField,
    ScriptEntry,
    scan_parameter_mapping,
    validate_selected_parameters,
)


def script(*names: str) -> tuple[ScriptEntry, ...]:
    return tuple(
        ScriptEntry(name=name, current_value=index + 10, order=index)
        for index, name in enumerate(names)
    )


def field(
    name: str,
    *,
    address: int,
    bit_start: int = 0,
    bits: int = 8,
    encoding_min: int = 0,
    encoding_max: int = 255,
) -> RegMapField:
    return RegMapField(
        name=name,
        channel0_address=address,
        channel_stride=0x100,
        bit_start=bit_start,
        bits=bits,
        encoding_min=encoding_min,
        encoding_max=encoding_max,
    )


def test_safe_parameter_names_are_exactly_the_approved_v11_list():
    assert SAFE_PARAMETER_NAMES == (
        "ibias_dac",
        "vref",
        "vbias_ref",
        "vga_cur",
        "res_gain",
        "res_bw",
        "gain_ctrl",
        "efin_cur",
        "eq",
    )


def test_scan_maps_safe_fields_and_preserves_script_order():
    entries = script("eq", "ibias_dac", "vref")
    fields = (
        field("vref", address=0x200B, bits=4, encoding_max=15),
        field("eq", address=0x2006, bits=5, encoding_max=31),
        field("ibias_dac", address=0x2002, bits=7, encoding_max=127),
    )

    mapping = scan_parameter_mapping(entries, fields)

    available = tuple(
        binding.name
        for binding in mapping.bindings
        if binding.status is MappingStatus.AVAILABLE
    )
    assert available == ("eq", "ibias_dac", "vref")
    eq = mapping.require_available("eq")
    assert eq.current_value == 10
    assert eq.script_order == 0
    assert eq.channel0_address == 0x2006
    assert eq.address_for_channel(2) == 0x2206
    assert eq.bit_start == 0
    assert eq.bits == 5
    assert eq.encoding_range == (0, 31)


def test_unsafe_control_and_test_fields_are_explicitly_excluded():
    entries = script(
        "testch",
        "power_up_done",
        "disable",
        "spare",
        "channel_enable",
        "eq",
    )
    fields = tuple(
        field(name, address=0x2000 + index)
        for index, name in enumerate(
            ("testch", "power_up_done", "disable", "spare", "channel_enable", "eq")
        )
    )

    mapping = scan_parameter_mapping(entries, fields)

    for name in ("testch", "power_up_done", "disable", "spare", "channel_enable"):
        excluded = mapping.binding(name)
        assert excluded.status is MappingStatus.EXCLUDED
        assert "safe list" in str(excluded.reason)
    assert mapping.require_available("eq").status is MappingStatus.AVAILABLE


def test_missing_and_ambiguous_fields_are_unavailable_without_hiding_usable_fields():
    entries = script("eq", "vref", "ibias_dac")
    fields = (
        field("eq", address=0x2006, bits=5, encoding_max=31),
        field("vref", address=0x200B, bits=4, encoding_max=15),
        field("VREF", address=0x210B, bits=4, encoding_max=15),
    )

    mapping = scan_parameter_mapping(entries, fields)

    assert mapping.require_available("eq").channel0_address == 0x2006
    assert mapping.binding("vref").status is MappingStatus.UNAVAILABLE
    assert "multiple RegMap" in str(mapping.binding("vref").reason)
    assert mapping.binding("ibias_dac").status is MappingStatus.UNAVAILABLE
    assert "missing from RegMap" in str(mapping.binding("ibias_dac").reason)
    assert mapping.binding("res_gain").status is MappingStatus.UNAVAILABLE
    assert "missing from Script" in str(mapping.binding("res_gain").reason)


def test_duplicate_script_entries_make_only_that_safe_parameter_unavailable():
    entries = (
        ScriptEntry("eq", 22, 0),
        ScriptEntry("eq", 23, 1),
        ScriptEntry("vref", 15, 2),
    )
    fields = (
        field("eq", address=0x2006, bits=5, encoding_max=31),
        field("vref", address=0x200B, bits=4, encoding_max=15),
    )

    mapping = scan_parameter_mapping(entries, fields)

    assert mapping.binding("eq").status is MappingStatus.UNAVAILABLE
    assert "multiple Script" in str(mapping.binding("eq").reason)
    assert mapping.require_available("vref").current_value == 15


def test_selected_unavailable_parameter_prevents_start():
    mapping = scan_parameter_mapping(
        script("eq", "vref"),
        (field("eq", address=0x2006, bits=5, encoding_max=31),),
    )

    with pytest.raises(ParameterSelectionError, match="vref.*UNAVAILABLE"):
        validate_selected_parameters(
            (ParameterSelection("vref", minimum=0, maximum=15, step=1),),
            mapping,
        )


def test_frontend_selection_cannot_supply_register_address_or_bitfield():
    valid = ParameterSelection.from_payload(
        {"name": "eq", "min": 0, "max": 31, "step": 2}
    )
    assert valid.name == "eq"

    with pytest.raises(ParameterSelectionError, match="unknown.*address"):
        ParameterSelection.from_payload(
            {"name": "eq", "min": 0, "max": 31, "step": 2, "address": 0x2006}
        )
    with pytest.raises(ParameterSelectionError, match="unknown.*bit"):
        ParameterSelection.from_payload(
            {"name": "eq", "min": 0, "max": 31, "step": 2, "bit": 0}
        )


def test_selection_rejects_hard_range_violation_and_never_appends_unaligned_max():
    mapping = scan_parameter_mapping(
        script("eq"),
        (field("eq", address=0x2006, bits=5, encoding_max=31),),
    )

    validated = validate_selected_parameters(
        (ParameterSelection("eq", minimum=1, maximum=10, step=4),), mapping
    )
    assert validated[0].values == (1, 5, 9)

    with pytest.raises(ParameterSelectionError, match="hard-coded.*range"):
        validate_selected_parameters(
            (ParameterSelection("eq", minimum=0, maximum=32, step=1),), mapping
        )


def test_validated_parameters_always_follow_script_order_not_request_order():
    mapping = scan_parameter_mapping(
        script("eq", "ibias_dac", "vref"),
        (
            field("eq", address=0x2006, bits=5, encoding_max=31),
            field("ibias_dac", address=0x2002, bits=7, encoding_max=127),
            field("vref", address=0x200B, bits=4, encoding_max=15),
        ),
    )

    validated = validate_selected_parameters(
        (
            ParameterSelection("vref", 10, 15, 1),
            ParameterSelection("eq", 16, 24, 2),
        ),
        mapping,
    )

    assert tuple(item.name for item in validated) == ("eq", "vref")


def test_mapping_hash_is_deterministic_and_covers_register_identity():
    entries = script("eq")
    original = scan_parameter_mapping(
        entries, (field("eq", address=0x2006, bits=5, encoding_max=31),)
    )
    repeated = scan_parameter_mapping(
        entries, (field("EQ", address=0x2006, bits=5, encoding_max=31),)
    )
    changed = scan_parameter_mapping(
        entries, (field("eq", address=0x2007, bits=5, encoding_max=31),)
    )

    assert original.content_hash == repeated.content_hash
    assert original.content_hash != changed.content_hash
