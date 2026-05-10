# Test Suite Documentation

**Last run:** 2026-05-01  
**Result:** 56 passed in 1.06s  
**Runner:** `.venv/bin/python -m pytest tests/ -v`

---

## test_bus.py — IPC Bus (14 tests)

Tests for `src/quadguide/core/bus.py`. The Bus is the shared-memory ring buffer that passes messages between processes. All state is created before `fork()` so workers inherit it without reconstruction.

### TestTopicRegistry

| Test | What it checks | Output |
|------|---------------|--------|
| `test_all_nine_topics_registered` | `Bus.__init__` pre-declares exactly the 9 expected topics (`kcf/estimate`, `nano/estimate`, `target/estimate`, `fc/attitude`, `fc/imu`, `guidance/accel`, `control/cmd`, `lockon/cmd`, `system/health`) — no more, no less | PASSED |
| `test_topics_constant_has_nine_entries` | The module-level `TOPICS` dict has exactly 9 entries | PASSED |

### TestLatest

| Test | What it checks | Output |
|------|---------------|--------|
| `test_returns_none_when_empty` | `bus.latest()` returns `None` on a topic that has never received a `publish()` — head sentinel is -1 | PASSED |
| `test_unknown_topic_raises` | `bus.latest("nonexistent/topic")` raises `KeyError` immediately — unknown topics are a programming error | PASSED |

### TestPublish

Float values in these tests are chosen to be exactly representable as IEEE 754 float32 so `==` survives the pack/unpack round-trip.

| Test | What it checks | Output |
|------|---------------|--------|
| `test_publish_then_latest_returns_message` | A message published to `guidance/accel` is returned verbatim by `latest()` | PASSED |
| `test_latest_returns_most_recent` | After two publishes, `latest()` returns the second (most recent) message, not the first | PASSED |
| `test_ring_wrap_returns_latest` | Publishing 5 messages into a `ring_depth=4` bus wraps the ring; `latest()` still returns the last message published | PASSED |
| `test_publish_unknown_topic_raises` | `bus.publish("nonexistent/topic", msg)` raises `KeyError` | PASSED |
| `test_different_topics_independent` | Publishing to `guidance/accel` and `control/cmd` does not cross-contaminate; each topic holds its own latest message | PASSED |

### TestSubscribeOne

| Test | What it checks | Output |
|------|---------------|--------|
| `test_blocks_until_publish` | `subscribe_one()` blocks for at least 40 ms while a background thread sleeps 50 ms then publishes — verifies pipe-backed blocking works and the returned message is correct | PASSED |
| `test_unknown_topic_raises` | `subscribe_one("nonexistent/topic")` raises `KeyError` before blocking | PASSED |

### TestSubscribeAny

| Test | What it checks | Output |
|------|---------------|--------|
| `test_wakes_on_first_publish` | Subscribing to two topics, a background thread publishes to `control/cmd` after 20 ms — `subscribe_any()` returns `("control/cmd", msg)` | PASSED |
| `test_unknown_topic_in_list_raises` | An unknown topic anywhere in the list raises `KeyError` before blocking | PASSED |

### TestDetach

| Test | What it checks | Output |
|------|---------------|--------|
| `test_detach_does_not_raise` | `bus.detach()` closes all shared memory and fd references without raising — does not call `shm.unlink()` (that belongs to the parent via `close()`) | PASSED |

---

## test_config.py — YAML Config Loader (17 tests)

Tests for `src/quadguide/core/config.py`. Covers loading `configs/config.yaml`, applying CLI-style overrides, required-section validation, and all 8 typed accessor functions.

### TestLoadConfig

| Test | What it checks | Output |
|------|---------------|--------|
| `test_loads_real_config` | `load_config("configs/config.yaml", {})` succeeds and returns a `dict` | PASSED |
| `test_all_required_sections_present` | The loaded config contains all 7 required top-level keys: `platform`, `airframe`, `tracker`, `guidance`, `watchdog`, `mission`, `logging` | PASSED |
| `test_missing_top_level_section_raises` | A YAML file with only `platform:` raises `KeyError` mentioning `airframe` | PASSED |
| `test_override_string_value` | Override `platform.name=dev_pc` sets the string value correctly | PASSED |
| `test_override_integer_value` | Override `platform.camera.width=320` coerces the string `"320"` to `int` `320` (type of the existing YAML value) | PASSED |
| `test_override_bool_value` | Override `platform.realtime.control_sched_fifo=false` sets the value to `False` — guards against `bool("false") == True` in Python | PASSED |
| `test_override_unknown_path_raises` | Overriding a dotpath that does not exist in the YAML raises `KeyError` | PASSED |

### TestAccessors

All accessor tests load `configs/config.yaml` once in `setup_method` and verify the typed dataclass fields match the expected values from the YAML file.

| Test | What it checks | Output |
|------|---------------|--------|
| `test_cfg_platform` | `cfg_platform()` returns a `PlatformConfig` with correct nested values: `name="orange_pi5"`, camera `width=640`, `fps=60`, serial `baud=115200`, inference `device="rknn"`, realtime `kcf_cpu_core=1`, `control_sched_fifo=True` | PASSED |
| `test_cfg_airframe` | `cfg_airframe()` returns correct `name`, `mass_kg` (approx 0.18), a 3-element `inertia` tuple, and `control_limits.max_roll_deg=35` | PASSED |
| `test_cfg_tracker` | `cfg_tracker()` returns KCF `detect_thresh≈0.5`, NanoTrack `exemplar_sz=127`, fusion `confidence_gate≈0.7` | PASSED |
| `test_cfg_guidance` | `cfg_guidance()` returns `N≈4.0`, `closing_vel_fallback≈2.0` | PASSED |
| `test_cfg_watchdog` | `cfg_watchdog()` returns `target_estimate_ms=150`, `fc_attitude_ms=50` | PASSED |
| `test_cfg_mission_with_hil` | `cfg_mission()` returns `mode="bench_hil"` and a non-None `hil` with `target_model="constant_velocity"` and a 3-element `initial_offset_m` | PASSED |
| `test_cfg_mission_hil_none_when_absent` | When the `mission` dict has no `hil` key, `cfg_mission().hil` is `None` | PASSED |
| `test_cfg_logging` | `cfg_logging()` returns `level="INFO"`, `max_bytes=10_485_760` | PASSED |
| `test_cfg_bus_from_config` | `cfg_bus()` reads `ring_depth=8` from the YAML `bus:` section | PASSED |
| `test_cfg_bus_defaults_when_section_absent` | `cfg_bus({})` (empty dict, no `bus:` key) returns `BusConfig(ring_depth=8)` — the default | PASSED |

---

## test_messages.py — Wire Message Types (25 tests)

Tests for `src/quadguide/core/messages.py`. Covers enum ordinal encoding, struct format byte sizes, and pack/unpack round-trips for all 8 message types.

### TestEnumOrdinals

| Test | What it checks | Output |
|------|---------------|--------|
| `test_tracker_health_round_trip` | Every `TrackerHealth` member round-trips through `_ord` → `_from_ord` and comes back equal | PASSED |
| `test_active_tracker_round_trip` | Same for `ActiveTracker` | PASSED |
| `test_process_state_round_trip` | Same for `ProcessState` | PASSED |
| `test_tracker_health_is_str` | `TrackerHealth.NOMINAL == "nominal"` — confirms `str` subclass behaviour for logging/JSON | PASSED |
| `test_active_tracker_is_str` | `ActiveTracker.CCV == "ccv"` | PASSED |
| `test_process_state_is_str` | `ProcessState.OK == "ok"` | PASSED |

### TestFormatSizes

Verifies `struct.calcsize(FMT_*)` matches the documented byte size. Format strings are the source of truth; these tests catch any accidental edits.

| Test | Format | Expected size | Output |
|------|--------|--------------|--------|
| `test_tracker_estimate` | `!QfffffB` | 29 bytes | PASSED |
| `test_target_estimate` | `!QfffffffBB` | 38 bytes | PASSED |
| `test_attitude_state` | `!Qffffff` | 32 bytes | PASSED |
| `test_imu_frame` | `!Qffffff` | 32 bytes | PASSED |
| `test_accel_cmd` | `!Qff` | 16 bytes | PASSED |
| `test_control_cmd` | `!Qffff` | 24 bytes | PASSED |
| `test_lockon_cmd` | `!QHffff` | 26 bytes | PASSED |
| `test_health_report` | `!Q16sB` | 25 bytes | PASSED |

### TestRoundTrips

All float fields use `pytest.approx(rel=1e-6)` because float fields are packed as IEEE 754 float32 (4 bytes), so round-trip equality is approximate — the dataclass stores Python float64 but the wire representation is float32. Integer and enum fields use exact `==`.

| Test | What it checks | Output |
|------|---------------|--------|
| `test_tracker_estimate` | `TrackerEstimate` pack/unpack: `timestamp_ns`, all 4 bbox floats, `confidence`, and `tracker_health` enum survive the round-trip | PASSED |
| `test_target_estimate` | `TargetEstimate` pack/unpack: bbox, `centroid_norm` tuple, `confidence`, `tracker_health`, and `active_tracker` enum | PASSED |
| `test_attitude_state` | `AttitudeState` pack/unpack: `timestamp_ns` and all 6 rate/angle floats | PASSED |
| `test_imu_frame` | `IMUFrame` pack/unpack: `timestamp_ns` and 6 IMU floats including `az=9.8` (not float32-exact; verified via approx) | PASSED |
| `test_accel_cmd` | `AccelCmd` pack/unpack: `timestamp_ns`, `ax`, `ay` | PASSED |
| `test_control_cmd` | `ControlCmd` pack/unpack: `timestamp_ns`, 4 control floats. `yaw_rate_dps=0.0` uses `abs=1e-9` tolerance (rel= against 0.0 resolves to zero tolerance) | PASSED |
| `test_lockon_cmd` | `LockOnCmd` pack/unpack: `timestamp_ns`, `seq=42` (exact), and bbox floats | PASSED |
| `test_lockon_cmd_seq_max` | `seq=65535` (uint16 max) survives pack/unpack without overflow or truncation | PASSED |
| `test_health_report_round_trip` | `HealthReport` pack/unpack: `timestamp_ns`, `process="camera"`, `state` enum survive; `detail="all good"` becomes `""` after unpack — detail is not on the wire | PASSED |
| `test_health_report_truncates_long_name` | A 20-character ASCII process name is silently truncated to 16 bytes without raising | PASSED |
| `test_health_report_truncates_multibyte_at_boundary` | `"☀" * 6` (18 UTF-8 bytes) is truncated to 5 copies (15 bytes) at a valid codepoint boundary — guards against slicing mid-codepoint which would produce `UnicodeDecodeError` at the receiver | PASSED |
