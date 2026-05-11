from __future__ import annotations
from dataclasses import dataclass, field
import yaml


# ── Leaf config dataclasses ──────────────────────────────────────────────────

@dataclass(frozen=True)
class BusConfig:
    ring_depth: int = 8


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
    guidance_accel_ms: int


@dataclass(frozen=True)
class GuidanceConfig:
    N: float
    closing_vel_fallback: float
    fov_horizontal_rad: float


@dataclass(frozen=True)
class FusionConfig:
    confidence_gate: float
    iou_divergence_thresh: float
    ncv_staleness_ms: int
    algorithm: str = "confidence_weighted"  # "confidence_weighted" | "iou_gated" | "passthrough"
    fast_tracker: str = "ccv"              # "ccv" | "ncv" — which tracker is the high-rate sync source
    iou_velocity_ema_alpha: float = 0.3    # iou_gated: EMA smoothing for velocity dead-reckoning
    iou_thresh_high: float = 0.7           # iou_gated: above this, light blend toward slow tracker
    iou_thresh_low: float = 0.3            # iou_gated: below this, use slow tracker directly


@dataclass(frozen=True)
class NanotrackConfig:
    exemplar_sz: int
    instance_sz: int
    score_threshold: float


@dataclass(frozen=True)
class KCFConfig:
    detect_thresh: float
    sigma: float
    lambda_: float


@dataclass(frozen=True)
class MOSSEConfig:
    pass  # OpenCV MOSSE exposes no tunable parameters


@dataclass(frozen=True)
class TrackerConfig:
    fusion: FusionConfig
    ccv: str | None = None          # "kcf" | "mosse" | None
    ncv: str | None = None          # "nanotrack" | None
    kcf: KCFConfig | None = None
    nanotrack: NanotrackConfig | None = None
    mosse: MOSSEConfig | None = None


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
    kcf_cpu_core: int
    control_cpu_core: int
    control_sched_fifo: bool
    control_fifo_prio: int


@dataclass(frozen=True)
class InferenceConfig:
    device: str
    backbone: str
    head: str


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baud: int


@dataclass(frozen=True)
class CameraConfig:
    backend: str
    pipeline: str
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class PlatformConfig:
    name: str
    camera: CameraConfig
    serial: SerialConfig
    inference: InferenceConfig
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
        ),
        serial=SerialConfig(port=p["serial"]["port"], baud=p["serial"]["baud"]),
        inference=InferenceConfig(
            device=p["inference"]["device"],
            backbone=p["inference"]["backbone"],
            head=p["inference"]["head"],
        ),
        realtime=RealtimeConfig(
            kcf_cpu_core=p["realtime"]["kcf_cpu_core"],
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
    kcf_raw = t.get("kcf")
    nt_raw = t.get("nanotrack")
    return TrackerConfig(
        fusion=FusionConfig(
            confidence_gate=t["fusion"]["confidence_gate"],
            iou_divergence_thresh=t["fusion"]["iou_divergence_thresh"],
            ncv_staleness_ms=t["fusion"]["ncv_staleness_ms"],
            algorithm=t["fusion"].get("algorithm", "confidence_weighted"),
            fast_tracker=t["fusion"].get("fast_tracker", "ccv"),
            iou_velocity_ema_alpha=t["fusion"].get("iou_velocity_ema_alpha", 0.3),
            iou_thresh_high=t["fusion"].get("iou_thresh_high", 0.7),
            iou_thresh_low=t["fusion"].get("iou_thresh_low", 0.3),
        ),
        ccv=t.get("ccv"),
        ncv=t.get("ncv"),
        kcf=KCFConfig(
            detect_thresh=kcf_raw["detect_thresh"],
            sigma=kcf_raw["sigma"],
            lambda_=kcf_raw["lambda_"],
        ) if kcf_raw else None,
        nanotrack=NanotrackConfig(
            exemplar_sz=nt_raw["exemplar_sz"],
            instance_sz=nt_raw["instance_sz"],
            score_threshold=nt_raw["score_threshold"],
        ) if nt_raw else None,
        mosse=MOSSEConfig() if "mosse" in t else None,
    )


def cfg_guidance(d: dict) -> GuidanceConfig:
    g = d["guidance"]
    return GuidanceConfig(
        N=g["N"],
        closing_vel_fallback=g["closing_vel_fallback"],
        fov_horizontal_rad=g["fov_horizontal_rad"],
    )


def cfg_watchdog(d: dict) -> WatchdogConfig:
    w = d["watchdog"]
    return WatchdogConfig(
        target_estimate_ms=w["target_estimate_ms"],
        fc_attitude_ms=w["fc_attitude_ms"],
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
