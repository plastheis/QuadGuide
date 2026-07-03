from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import yaml


# ── Leaf config dataclasses ──────────────────────────────────────────────────

@dataclass(frozen=True)
class BusConfig:
    ring_depth: int = 8


@dataclass(frozen=True)
class DiagConfig:
    trace: bool = False              # write a post-run latency/state trace (enabled per-run via --log)
    trace_dir: str | None = None     # destination dir; resolved by run.py when --log is set
    trace_max_rows: int = 0          # per-process row cap; 0 = unbounded


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    dir: str
    max_bytes: int
    backup_count: int


@dataclass(frozen=True)
class HILConfig:
    target_model: str
    initial_offset_m: tuple[float, float, float]
    target_speed_mps: float


@dataclass(frozen=True)
class MissionConfig:
    mode: str
    hil: HILConfig | None = None


@dataclass(frozen=True)
class WatchdogConfig:
    target_estimate_ms: int
    fc_attitude_ms: int
    fc_imu_ms: int
    guidance_accel_ms: int


@dataclass(frozen=True)
class PronavConfig:
    N: float
    closing_vel_fallback: float
    closing_vel_ema_alpha: float = 0.3
    closing_vel_min_area_rate: float = 0.001
    closing_vel_area_scale: float = 5.0


@dataclass(frozen=True)
class PurePursuitConfig:
    K: float                # m/s² per radian of LOS angle
    deadband: float = 0.03  # centroid half-width (frac of half-FoV) zeroed near boresight


@dataclass(frozen=True)
class GuidanceConfig:
    method: str             # "pronav" | "pure_pursuit"
    fov_horizontal_rad: float
    throttle_hold: float = 0.55
    pronav: PronavConfig | None = None
    pure_pursuit: PurePursuitConfig | None = None


@dataclass(frozen=True)
class TrackerConfig:
    import_spec: str                                          # YAML key: "import"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlLimitsConfig:
    max_roll_deg: float
    max_pitch_deg: float
    max_roll_rate_dps: float
    max_pitch_rate_dps: float


@dataclass(frozen=True)
class AirframeConfig:
    name: str
    mass_kg: float
    inertia: tuple[float, float, float]
    control_limits: ControlLimitsConfig


@dataclass(frozen=True)
class RealtimeConfig:
    tracker_cpu_core: int | None
    control_cpu_core: int
    control_sched_fifo: bool
    control_fifo_prio: int


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baud: int
    mode: str = "uart"          # "uart" | "tcp" (HIL); selects the link transport
    tcp_host: str = ""          # used when mode=tcp — dev-machine ArduPilot SITL host
    tcp_port: int = 0           # used when mode=tcp — dev-machine ArduPilot SITL port


@dataclass(frozen=True)
class CameraConfig:
    backend: str
    pipeline: str
    width: int
    height: int
    fps: int
    url: str = ""               # used when backend=network (HIL) — MJPEG stream URL
    raw_tcp_host: str = ""      # used when backend=raw_tcp (HIL) — dev-machine frame server host
    raw_tcp_port: int = 8091    # used when backend=raw_tcp (HIL) — dev-machine frame server port
    device: str = ""            # used when backend=csi — V4L2 capture node (default /dev/video0)
    media: str = ""             # used when backend=csi — media device for pad setup (default /dev/media0)
    gain: int = 0               # used when backend=csi — OV9281 analogue_gain (16..248); 0 = leave sensor default
    exposure: int = 0           # used when backend=csi — OV9281 exposure in lines (4..3652); 0 = leave sensor default
    auto_exposure: bool = True  # used when backend=csi — software AEC (no ISP on raw path); gain/exposure seed the loop
    ae_target: int = 210        # used when backend=csi — AE setpoint: 95th-pctl brightness held just below clip


@dataclass(frozen=True)
class PlatformConfig:
    name: str
    camera: CameraConfig
    serial: SerialConfig
    realtime: RealtimeConfig


# ── Loader ───────────────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = frozenset(
    {"platform", "airframe", "tracker", "guidance", "watchdog", "mission", "logging"}
)


def load_config(path: str, overrides: dict[str, str]) -> dict:
    """Load YAML config, apply dot-notation overrides, validate required sections."""
    with open(path) as f:
        config = yaml.safe_load(f)

    for dotpath, str_value in overrides.items():
        parts = dotpath.split(".")
        node = config
        for part in parts[:-1]:
            node = node[part]  # KeyError propagates if path is wrong
        leaf_key = parts[-1]
        existing = node[leaf_key]   # KeyError if leaf doesn't exist
        if isinstance(existing, bool):
            node[leaf_key] = str_value.lower() in ("1", "true", "yes")
        else:
            node[leaf_key] = type(existing)(str_value)

    missing = _REQUIRED_SECTIONS - config.keys()
    if missing:
        raise KeyError(f"Required config section(s) missing: {sorted(missing)}")

    return config


# ── Typed accessors ──────────────────────────────────────────────────────────

def cfg_platform(d: dict) -> PlatformConfig:
    p = d["platform"]
    cam = p["camera"]
    return PlatformConfig(
        name=p["name"],
        camera=CameraConfig(
            backend=cam["backend"],
            pipeline=cam.get("pipeline", ""),
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
            url=cam.get("url", ""),
            raw_tcp_host=cam.get("raw_tcp_host", ""),
            raw_tcp_port=cam.get("raw_tcp_port", 8091),
            device=cam.get("device", ""),
            media=cam.get("media", ""),
            gain=cam.get("gain", 0),
            exposure=cam.get("exposure", 0),
            auto_exposure=cam.get("auto_exposure", True),
            ae_target=cam.get("ae_target", 210),
        ),
        serial=SerialConfig(
            # port/baud are irrelevant in tcp (HIL) mode — tolerate their absence.
            port=p["serial"].get("port", ""),
            baud=p["serial"].get("baud", 0),
            mode=p["serial"].get("mode", "uart"),
            tcp_host=p["serial"].get("tcp_host", ""),
            tcp_port=p["serial"].get("tcp_port", 0),
        ),
        realtime=RealtimeConfig(
            tracker_cpu_core=p["realtime"].get("tracker_cpu_core"),
            control_cpu_core=p["realtime"]["control_cpu_core"],
            control_sched_fifo=p["realtime"]["control_sched_fifo"],
            control_fifo_prio=p["realtime"]["control_fifo_prio"],
        ),
    )


def cfg_airframe(d: dict) -> AirframeConfig:
    a = d["airframe"]
    lim = a["control_limits"]
    return AirframeConfig(
        name=a["name"],
        mass_kg=a["mass_kg"],
        inertia=tuple(a["inertia"]),
        control_limits=ControlLimitsConfig(
            max_roll_deg=lim["max_roll_deg"],
            max_pitch_deg=lim["max_pitch_deg"],
            max_roll_rate_dps=lim["max_roll_rate_dps"],
            max_pitch_rate_dps=lim["max_pitch_rate_dps"],
        ),
    )


def cfg_tracker(d: dict) -> TrackerConfig:
    t = d["tracker"]
    return TrackerConfig(
        import_spec=t["import"],
        params=dict(t.get("params") or {}),
    )


def cfg_guidance(d: dict) -> GuidanceConfig:
    g = d["guidance"]
    pn_raw = g.get("pronav")
    pp_raw = g.get("pure_pursuit")
    return GuidanceConfig(
        method=g["method"],
        fov_horizontal_rad=g["fov_horizontal_rad"],
        throttle_hold=g.get("throttle_hold", 0.55),
        pronav=PronavConfig(
            N=pn_raw["N"],
            closing_vel_fallback=pn_raw["closing_vel_fallback"],
            closing_vel_ema_alpha=pn_raw.get("closing_vel_ema_alpha", 0.3),
            closing_vel_min_area_rate=pn_raw.get("closing_vel_min_area_rate", 0.001),
            closing_vel_area_scale=pn_raw.get("closing_vel_area_scale", 5.0),
        ) if pn_raw else None,
        pure_pursuit=PurePursuitConfig(
            K=pp_raw["K"],
            deadband=pp_raw.get("deadband", 0.03),
        ) if pp_raw else None,
    )


def cfg_watchdog(d: dict) -> WatchdogConfig:
    w = d["watchdog"]
    return WatchdogConfig(
        target_estimate_ms=w["target_estimate_ms"],
        fc_attitude_ms=w["fc_attitude_ms"],
        fc_imu_ms=w["fc_imu_ms"],
        guidance_accel_ms=w["guidance_accel_ms"],
    )


def cfg_mission(d: dict) -> MissionConfig:
    m = d["mission"]
    hil_raw = m.get("hil")
    hil = None
    if hil_raw is not None:
        hil = HILConfig(
            target_model=hil_raw["target_model"],
            initial_offset_m=tuple(hil_raw["initial_offset_m"]),
            target_speed_mps=hil_raw["target_speed_mps"],
        )
    return MissionConfig(mode=m["mode"], hil=hil)


def cfg_logging(d: dict) -> LoggingConfig:
    lg = d["logging"]
    return LoggingConfig(
        level=lg["level"],
        dir=lg["dir"],
        max_bytes=lg["max_bytes"],
        backup_count=lg["backup_count"],
    )


def cfg_bus(d: dict) -> BusConfig:
    bus_raw = d.get("bus", {})
    return BusConfig(ring_depth=bus_raw.get("ring_depth", 8))


def cfg_diag(d: dict) -> DiagConfig:
    diag_raw = d.get("diag") or {}
    return DiagConfig(
        trace=diag_raw.get("trace", False),
        trace_dir=diag_raw.get("trace_dir"),
        trace_max_rows=diag_raw.get("trace_max_rows", 0),
    )
