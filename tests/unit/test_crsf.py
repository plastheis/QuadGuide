import struct
import pytest
from quadguide.link.crsf import (
    crc8, build_frame, pack_channels,
    CRSFFrame, CRSF_SYNC, CRSF_ATTITUDE, CRSF_RC_CHANNELS,
)


# --- CRC8 ---

def _ref_crc8(data: bytes) -> int:
    """Reference implementation to validate against."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def test_crc8_matches_reference_single_byte():
    data = bytes([0x1E])
    assert crc8(data) == _ref_crc8(data)


def test_crc8_matches_reference_multi_byte():
    data = bytes([0x16]) + bytes(22)
    assert crc8(data) == _ref_crc8(data)


def test_crc8_different_inputs_differ():
    assert crc8(b'\x1e\x00\x00') != crc8(b'\x1e\x00\x01')


# --- build_frame ---

def test_build_frame_sync_byte():
    frame = build_frame(CRSF_ATTITUDE, bytes(6))
    assert frame[0] == CRSF_SYNC  # 0xC8


def test_build_frame_length_field():
    # len = type(1) + payload_len + crc(1)
    frame = build_frame(CRSF_ATTITUDE, bytes(6))
    assert frame[1] == 8   # 1 + 6 + 1


def test_build_frame_type_byte():
    frame = build_frame(CRSF_ATTITUDE, bytes(6))
    assert frame[2] == CRSF_ATTITUDE


def test_build_frame_total_length():
    # sync(1) + len(1) + type(1) + payload(6) + crc(1) = 10
    frame = build_frame(CRSF_ATTITUDE, bytes(6))
    assert len(frame) == 10


def test_build_frame_rc_channels_length():
    # sync(1) + len(1) + type(1) + payload(22) + crc(1) = 26
    frame = build_frame(CRSF_RC_CHANNELS, bytes(22))
    assert len(frame) == 26
    assert frame[1] == 24  # 1 + 22 + 1


def test_build_frame_crc_appended():
    payload = struct.pack(">hhh", 100, 200, 300)
    frame = build_frame(CRSF_ATTITUDE, payload)
    expected_crc = _ref_crc8(bytes([CRSF_ATTITUDE]) + payload)
    assert frame[-1] == expected_crc


# --- pack_channels ---

def test_pack_channels_produces_22_bytes():
    assert len(pack_channels([992] * 16)) == 22


def test_pack_channels_center_values():
    packed = pack_channels([992] * 16)
    bits = int.from_bytes(packed, "little")
    for i in range(16):
        assert (bits >> (i * 11)) & 0x7FF == 992


def test_pack_channels_min_max():
    channels = [172, 1811] + [992] * 14
    packed = pack_channels(channels)
    bits = int.from_bytes(packed, "little")
    assert (bits >> 0) & 0x7FF == 172    # ch1 min
    assert (bits >> 11) & 0x7FF == 1811  # ch2 max


def test_pack_channels_all_independent():
    # Each channel occupies its own 11-bit slot; changing ch3 should not affect ch1
    base = [992] * 16
    modified = list(base)
    modified[2] = 500
    packed_base = pack_channels(base)
    packed_mod  = pack_channels(modified)
    bits_base = int.from_bytes(packed_base, "little")
    bits_mod  = int.from_bytes(packed_mod, "little")
    assert (bits_base >> 0) & 0x7FF == (bits_mod >> 0) & 0x7FF  # ch1 unchanged
    assert (bits_mod >> 22) & 0x7FF == 500                       # ch3 changed
