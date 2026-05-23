from __future__ import annotations
import time
from dataclasses import dataclass
from enum import IntEnum

CRSF_SYNC         = 0xC8
CRSF_GPS          = 0x02
CRSF_BATTERY      = 0x08
CRSF_RC_CHANNELS  = 0x16
CRSF_ATTITUDE     = 0x1E
CRSF_FLIGHT_MODE  = 0x21
CRSF_IMU_RAW      = 0x80   # custom madflight frame: 6×int16 (ax,ay,az,gx,gy,gz)

_MAX_LEN = 62  # max valid len field value (payload ≤ 60, +type+crc = 62)

# CRSF tick range corresponds to 988–2012 µs pulse width.
_TICK_MIN = 172
_TICK_MAX = 1811
_US_MIN   = 988.0
_US_MAX   = 2012.0


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


def us_to_ticks(us: float) -> int:
    """Convert µs pulse width to CRSF 11-bit ticks. Clamps to [172, 1811]."""
    ticks = (us - 1500.0) * 8.0 / 5.0 + 992.0
    return int(max(_TICK_MIN, min(_TICK_MAX, ticks)))


def ticks_to_us(ticks: int) -> float:
    """Convert CRSF 11-bit ticks to µs pulse width."""
    return (ticks - 992) * 5.0 / 8.0 + 1500.0


@dataclass
class CRSFFrame:
    type: int
    payload: bytes
    timestamp_ns: int


def build_frame(frame_type: int, payload: bytes) -> bytes:
    length = len(payload) + 2  # type(1) + crc(1)
    crc_input = bytes([frame_type]) + payload
    return bytes([CRSF_SYNC, length, frame_type]) + payload + bytes([crc8(crc_input)])


def pack_channels(channels_us: list[float]) -> bytes:
    """Pack 16 channel µs values into 22 bytes (16 × 11-bit CRSF, LSB-first)."""
    assert len(channels_us) == 16
    bits = 0
    for i, us in enumerate(channels_us):
        bits |= (us_to_ticks(us) & 0x7FF) << (i * 11)
    return bits.to_bytes(22, "little")


class _State(IntEnum):
    WAIT_SYNC    = 0
    READ_LEN     = 1
    READ_TYPE    = 2
    READ_PAYLOAD = 3
    READ_CRC     = 4


class CRSFParser:
    def __init__(self):
        self._state     = _State.WAIT_SYNC
        self._len       = 0
        self._type      = 0
        self._payload   = bytearray()
        self._remaining = 0

    def feed(self, byte: int) -> CRSFFrame | None:
        if self._state == _State.WAIT_SYNC:
            if byte == CRSF_SYNC:
                self._state = _State.READ_LEN

        elif self._state == _State.READ_LEN:
            if byte < 2 or byte > _MAX_LEN:
                self._reset()
            else:
                self._len       = byte
                self._remaining = byte - 2  # payload bytes = len - type(1) - crc(1)
                self._state     = _State.READ_TYPE

        elif self._state == _State.READ_TYPE:
            self._type    = byte
            self._payload = bytearray()
            self._state   = _State.READ_PAYLOAD if self._remaining > 0 else _State.READ_CRC

        elif self._state == _State.READ_PAYLOAD:
            self._payload.append(byte)
            self._remaining -= 1
            if self._remaining == 0:
                self._state = _State.READ_CRC

        elif self._state == _State.READ_CRC:
            self._state = _State.WAIT_SYNC
            expected = crc8(bytes([self._type]) + self._payload)
            if byte == expected:
                return CRSFFrame(
                    type=self._type,
                    payload=bytes(self._payload),
                    timestamp_ns=time.monotonic_ns(),
                )
            # CRC mismatch — drop frame silently

        return None

    def _reset(self) -> None:
        self._state   = _State.WAIT_SYNC
        self._payload = bytearray()
