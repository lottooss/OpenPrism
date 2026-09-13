# tests/test_usb_hid_actuator.py
"""Python test suite for Generic USB HID actuator protocol & configuration (Milestone M9-03)."""

from __future__ import annotations

from pathlib import Path
import struct
import yaml
import jsonschema

# CCITT-16 CRC (poly 0x1021, init 0xFFFF) matching C++ compute_usb_hid_crc16
def compute_crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def test_usb_hid_config_conforms_to_schema() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    schema_path = repo_root / "schemas" / "config" / "actuator.schema.json"
    config_path = repo_root / "configs" / "actuator" / "usb_hid.yaml"

    assert schema_path.is_file(), f"Missing schema at {schema_path}"
    assert config_path.is_file(), f"Missing config at {config_path}"

    import json
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    assert "actuator" in config_data, "Config must contain top-level 'actuator' key"
    actuator_config = config_data["actuator"]

    # Validate against JSON Schema
    jsonschema.validate(instance=actuator_config, schema=schema)
    assert actuator_config["backend"] == "usb_hid"
    assert actuator_config["hid_com_port"] == "COM3"
    assert actuator_config["hid_baud_rate"] == 115200


def test_usb_hid_binary_protocol_structure() -> None:
    # Wire protocol layout:
    # 2s (magic 0x5548 = 'UH' little endian, or 0x4855 uint16)
    # B (version = 1)
    # B (msg_type = 1)
    # Q (sequence_id = 100)
    # q (timestamp_ns = 1700000000)
    # i (delta_x = 25)
    # i (delta_y = -30)
    # B (button_action = 1)
    # B (mouse_button = 1)
    # B (held_buttons = 2)
    # B (flags = 0)
    # H (crc16)
    # Format: '<HBBQqiiBBBBH' -> 2+1+1+8+8+4+4+1+1+1+1+2 = 34 bytes
    fmt = "<HBBQqiiBBBB"
    payload = struct.pack(
        fmt,
        0x5548,       # magic
        1,            # version
        1,            # msg_type (command)
        100,          # sequence_id
        1700000000,   # timestamp_ns
        25,           # delta_x_counts
        -30,          # delta_y_counts
        1,            # button_action (press)
        1,            # mouse_button (left)
        2,            # held_buttons_mask
        0             # flags
    )
    assert len(payload) == 32

    crc = compute_crc16_ccitt(payload)
    full_packet = payload + struct.pack("<H", crc)
    assert len(full_packet) == 34

    # Verify unpack
    unpacked_crc = struct.unpack("<H", full_packet[32:34])[0]
    assert unpacked_crc == crc
    computed = compute_crc16_ccitt(full_packet[:32])
    assert computed == crc


def test_usb_hid_crc16_bit_corruption_detection() -> None:
    payload = b"\x48\x55\x01\x01" + b"\x00" * 28
    crc = compute_crc16_ccitt(payload)
    valid_packet = bytearray(payload + struct.pack("<H", crc))

    # Single bit corruption in payload
    valid_packet[10] ^= 0x01
    corrupted_crc = struct.unpack("<H", valid_packet[32:34])[0]
    recalculated = compute_crc16_ccitt(bytes(valid_packet[:32]))
    assert recalculated != corrupted_crc
