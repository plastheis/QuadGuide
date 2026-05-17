import struct
import pytest
from quadguide.link.crsf import (
    crc8, build_frame, pack_channels,
    CRSFFrame, CRSF_SYNC, CRSF_ATTITUDE, CRSF_RC_CHANNELS,
    CRSFParser,
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
    assert len(pack_channels([1500.0] * 16)) == 22


def test_pack_channels_center_values():
    # 1500µs = center = 992 ticks
    packed = pack_channels([1500.0] * 16)
    bits = int.from_bytes(packed, "little")
    for i in range(16):
        assert (bits >> (i * 11)) & 0x7FF == 992


def test_pack_channels_min_max():
    # 988µs = 172 ticks (CRSF min), 2012µs = 1811 ticks (CRSF max)
    channels = [988.0, 2012.0] + [1500.0] * 14
    packed = pack_channels(channels)
    bits = int.from_bytes(packed, "little")
    assert (bits >> 0) & 0x7FF == 172    # ch1 min
    assert (bits >> 11) & 0x7FF == 1811  # ch2 max


def test_pack_channels_all_independent():
    # Each channel occupies its own 11-bit slot; changing ch3 should not affect ch1.
    # 1100µs = 352 ticks.
    base     = [1500.0] * 16
    modified = list(base)
    modified[2] = 1100.0
    packed_base = pack_channels(base)
    packed_mod  = pack_channels(modified)
    bits_base = int.from_bytes(packed_base, "little")
    bits_mod  = int.from_bytes(packed_mod, "little")
    assert (bits_base >> 0) & 0x7FF == (bits_mod >> 0) & 0x7FF  # ch1 unchanged
    assert (bits_mod >> 22) & 0x7FF == 352                       # ch3 = 1100µs


# --- CRSFParser ---

class TestCRSFParser:
    def _feed_all(self, parser: CRSFParser, data: bytes) -> list[CRSFFrame]:
        results = []
        for b in data:
            frame = parser.feed(b)
            if frame is not None:
                results.append(frame)
        return results

    def test_parses_attitude_frame(self):
        payload = struct.pack(">hhh", 100, 200, 300)
        frame_bytes = build_frame(CRSF_ATTITUDE, payload)
        parser = CRSFParser()
        frames = self._feed_all(parser, frame_bytes)
        assert len(frames) == 1
        assert frames[0].type == CRSF_ATTITUDE
        assert frames[0].payload == payload

    def test_parses_rc_channels_frame(self):
        payload = pack_channels([1500.0] * 16)
        frame_bytes = build_frame(CRSF_RC_CHANNELS, payload)
        parser = CRSFParser()
        frames = self._feed_all(parser, frame_bytes)
        assert len(frames) == 1
        assert frames[0].type == CRSF_RC_CHANNELS

    def test_returns_none_until_frame_complete(self):
        payload = struct.pack(">hhh", 0, 0, 0)
        frame_bytes = build_frame(CRSF_ATTITUDE, payload)
        parser = CRSFParser()
        # All but the last byte should return None
        for b in frame_bytes[:-1]:
            assert parser.feed(b) is None
        # Last byte completes the frame
        assert parser.feed(frame_bytes[-1]) is not None

    def test_ignores_non_sync_bytes(self):
        parser = CRSFParser()
        for b in [0x00, 0x01, 0xFF, 0xAA, 0x42]:
            assert parser.feed(b) is None

    def test_rejects_crc_mismatch(self):
        payload = struct.pack(">hhh", 100, 200, 300)
        frame_bytes = bytearray(build_frame(CRSF_ATTITUDE, payload))
        frame_bytes[-1] ^= 0xFF  # corrupt CRC
        parser = CRSFParser()
        frames = self._feed_all(parser, bytes(frame_bytes))
        assert len(frames) == 0

    def test_recovers_after_crc_mismatch(self):
        # Bad frame followed by a good frame — parser must recover
        payload = struct.pack(">hhh", 100, 200, 300)
        bad = bytearray(build_frame(CRSF_ATTITUDE, payload))
        bad[-1] ^= 0xFF
        good = build_frame(CRSF_ATTITUDE, payload)
        parser = CRSFParser()
        frames = self._feed_all(parser, bytes(bad) + good)
        assert len(frames) == 1

    def test_parses_two_consecutive_frames(self):
        payload = struct.pack(">hhh", 1, 2, 3)
        frame_bytes = build_frame(CRSF_ATTITUDE, payload) * 2
        parser = CRSFParser()
        frames = self._feed_all(parser, frame_bytes)
        assert len(frames) == 2

    def test_resets_on_oversized_length(self):
        # Feed a sync byte followed by an invalid length (> 62)
        parser = CRSFParser()
        assert parser.feed(CRSF_SYNC) is None
        assert parser.feed(63) is None  # invalid len, should reset
        # Next valid frame should still parse
        payload = struct.pack(">hhh", 0, 0, 0)
        frames = self._feed_all(parser, build_frame(CRSF_ATTITUDE, payload))
        assert len(frames) == 1

    def test_frame_has_timestamp(self):
        payload = struct.pack(">hhh", 0, 0, 0)
        frame_bytes = build_frame(CRSF_ATTITUDE, payload)
        parser = CRSFParser()
        frames = self._feed_all(parser, frame_bytes)
        assert frames[0].timestamp_ns > 0
