from __future__ import annotations
import time
from dataclasses import dataclass
from enum import IntEnum

CRSF_SYNC        = 0xC8
CRSF_ATTITUDE    = 0x1E
CRSF_RC_CHANNELS = 0x16

_MAX_LEN = 62  # max valid len field value (payload ≤ 60, +type+crc = 62)


def _make_crc8_table(poly: int) -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
        table.append(crc)
    return table


_CRC8_TABLE = _make_crc8_table(0xD5)


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[crc ^ b]
    return crc


@dataclass
class CRSFFrame:
    type: int
    payload: bytes
    timestamp_ns: int


def build_frame(frame_type: int, payload: bytes) -> bytes:
    length = len(payload) + 2  # type(1) + crc(1)
    crc_input = bytes([frame_type]) + payload
    return bytes([CRSF_SYNC, length, frame_type]) + payload + bytes([crc8(crc_input)])


def pack_channels(channels: list[int]) -> bytes:
    assert len(channels) == 16
    bits = 0
    for i, ch in enumerate(channels):
        bits |= (ch & 0x7FF) << (i * 11)
    return bits.to_bytes(22, "little")
