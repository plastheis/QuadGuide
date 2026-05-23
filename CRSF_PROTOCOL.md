# CRSF Protocol — madflight UART Link Reference

This document describes all CRSF frames active on the madflight UART link. Use it to implement a parser on the receiving end.

---

## Physical Layer

| Parameter | Value |
|-----------|-------|
| Baud rate | 420 000 |
| Data bits | 8 |
| Stop bits | 1 |
| Parity    | None |
| Byte order | Big-endian |

---

## Frame Structure

Every frame (uplink and downlink) follows the same envelope:

```
Byte 0      : Device address / sync byte  (0xC8)
Byte 1      : Length  — number of bytes that follow: type(1) + payload(N) + crc(1)
Byte 2      : Frame type  (see table below)
Bytes 3…N+2 : Payload
Byte N+3    : CRC-8 DVB-S2  (covers bytes 2…N+2, i.e. type + payload)
```

Maximum total frame size: 64 bytes.

### CRC-8 DVB-S2

Polynomial `0xD5`, initial value `0x00`. Covers all bytes from the **type byte** to the last payload byte (does **not** include address or length).

```c
uint8_t crc8_dvb_s2(uint8_t crc, uint8_t byte) {
    crc ^= byte;
    for (int i = 0; i < 8; i++) {
        crc = (crc & 0x80) ? (crc << 1) ^ 0xD5 : (crc << 1);
    }
    return crc;
}
```

---

## Frame Type Summary

| Direction | Type byte | Name               | Payload (bytes) | Total frame (bytes) | Rate     |
|-----------|-----------|--------------------|-----------------|---------------------|----------|
| Uplink    | `0x16`    | RC Channels Packed | 22              | 26                  | Radio-rate (up to 500 Hz) |
| Downlink  | `0x02`    | GPS                | 15              | 19                  | 2 Hz     |
| Downlink  | `0x08`    | Battery Sensor     | 8               | 12                  | 2 Hz     |
| Downlink  | `0x1E`    | Attitude           | 6               | 10                  | 50 Hz    |
| Downlink  | `0x21`    | Flight Mode        | 1–16            | 5–20                | 50 Hz    |
| Downlink  | `0x80`    | **IMU Raw**        | 12              | 16                  | **50 Hz** |

---

## Uplink Frames (FC ← Radio Receiver)

### `0x16` RC Channels Packed

16 RC channels packed into 22 bytes, 11 bits per channel, little-endian bit order.

```
Byte 0  : 0xC8  sync
Byte 1  : 0x18  length = 24 (type + 22 payload + crc)
Byte 2  : 0x16  type
Bytes 3–24 : 22 payload bytes  (see below)
Byte 25 : CRC
```

**Payload bit layout** — all 16 channels are concatenated as 11-bit values, LSB first:

```
channel[0]  = bits  0–10
channel[1]  = bits 11–21
channel[2]  = bits 22–32
channel[3]  = bits 33–43
channel[4]  = bits 44–54
channel[5]  = bits 55–65
channel[6]  = bits 66–76
channel[7]  = bits 77–87
channel[8]  = bits 88–98
channel[9]  = bits 99–109
channel[10] = bits 110–120
channel[11] = bits 121–131
channel[12] = bits 132–142
channel[13] = bits 143–153
channel[14] = bits 154–164
channel[15] = bits 165–175
```

**Raw value → pulse width conversion:**

```
PWM_us = 988 + (raw - 172) * (2012 - 988) / (1811 - 172)
```

| Raw value | PWM [µs] |
|-----------|----------|
| 172       | 988 (min) |
| 992       | 1500 (mid) |
| 1811      | 2012 (max) |

---

## Downlink Frames (FC → Radio Receiver / Ground Station)

All downlink frames are emitted at a 50 Hz base rate (every 20 ms). Slower frames (Battery, GPS) are interleaved at 2 Hz.

---

### `0x02` GPS

Rate: 2 Hz

```
Byte 0  : 0xC8  sync
Byte 1  : 0x11  length = 17
Byte 2  : 0x02  type
Bytes 3–6   : int32_t  latitude        (degrees × 10 000 000)
Bytes 7–10  : int32_t  longitude       (degrees × 10 000 000)
Bytes 11–12 : uint16_t groundspeed     (km/h × 10)
Bytes 13–14 : uint16_t heading         (degrees × 100)
Bytes 15–16 : uint16_t altitude        (metres + 1000 m offset)
Byte 17     : uint8_t  satellites      (count)
Byte 18     : CRC
```

All multi-byte fields are big-endian.

---

### `0x08` Battery Sensor

Rate: 2 Hz

```
Byte 0  : 0xC8  sync
Byte 1  : 0x0A  length = 10
Byte 2  : 0x08  type
Bytes 3–4  : uint16_t  voltage         (V × 10)
Bytes 5–6  : uint16_t  current         (A × 10)
Bytes 7–9  : uint24_t  fuel            (drawn mAh, 24-bit)
Byte 10    : uint8_t   battery_remain  (percent 0–100)
Byte 11    : CRC
```

---

### `0x1E` Attitude

Rate: 50 Hz

```
Byte 0  : 0xC8  sync
Byte 1  : 0x08  length = 8
Byte 2  : 0x1E  type
Bytes 3–4 : int16_t  pitch   (radians × 10 000)
Bytes 5–6 : int16_t  roll    (radians × 10 000)
Bytes 7–8 : int16_t  yaw     (radians × 10 000)
Byte 9    : CRC
```

Angles are clamped to ±180°. To recover degrees: `degrees = raw / 174.532925`.

---

### `0x21` Flight Mode

Rate: 50 Hz

```
Byte 0  : 0xC8  sync
Byte 1  : length  (2 + string_len, where string_len ≤ 16)
Byte 2  : 0x21  type
Bytes 3…N : null-terminated ASCII string (max 16 bytes including null)
Byte N+1  : CRC
```

**String format:** `[*][sat_count]<mode_name>`

- Leading `*` when armed
- Satellite count prefixed when > 0
- `<mode_name>` is the vehicle flight mode string

Examples: `ANGLE`, `*ACRO`, `*12POSHOLD`

---

### `0x80` IMU Raw  *(custom frame)*

Rate: 50 Hz

```
Byte 0  : 0xC8  sync
Byte 1  : 0x0E  length = 14
Byte 2  : 0x80  type
Bytes 3–4   : int16_t  ax   North acceleration  (G × 1000,   i.e. milli-G)
Bytes 5–6   : int16_t  ay   East  acceleration  (G × 1000)
Bytes 7–8   : int16_t  az   Down  acceleration  (G × 1000)
Bytes 9–10  : int16_t  gx   North rotation rate (deg/s × 10)
Bytes 11–12 : int16_t  gy   East  rotation rate (deg/s × 10)
Bytes 13–14 : int16_t  gz   Down  rotation rate (deg/s × 10)
Byte 15     : CRC
```

All fields are signed, big-endian.

**To recover physical values:**

```c
float ax_G     = (float)raw_ax / 1000.0f;   // G
float ay_G     = (float)raw_ay / 1000.0f;
float az_G     = (float)raw_az / 1000.0f;
float gx_dps   = (float)raw_gx / 10.0f;     // deg/s
float gy_dps   = (float)raw_gy / 10.0f;
float gz_dps   = (float)raw_gz / 10.0f;
```

**Coordinate frame:** NED (North-East-Down), body-fixed axes.

| Field | Full-scale range        | Resolution  |
|-------|-------------------------|-------------|
| ax    | ±32.767 G               | 0.001 G     |
| ay    | ±32.767 G               | 0.001 G     |
| az    | ±32.767 G               | 0.001 G     |
| gx    | ±3276.7 deg/s           | 0.1 deg/s   |
| gy    | ±3276.7 deg/s           | 0.1 deg/s   |
| gz    | ±3276.7 deg/s           | 0.1 deg/s   |

---

## Parser Notes

- Sync byte is always `0xC8`. Use it to re-align after a framing error.
- Read byte 1 (length), then read exactly `length` more bytes; the last byte is the CRC.
- Verify CRC before processing. Discard the frame silently on mismatch.
- Frame type `0x80` will not be recognised by standard EdgeTX/OpenTX telemetry sensors; use a LUA script or external parser to consume it.
