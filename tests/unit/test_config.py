import os
import pathlib
import tempfile
import pytest
from quadguide.core.config import (
    load_config,
    cfg_platform, cfg_airframe, cfg_tracker,
    cfg_guidance, cfg_watchdog, cfg_mission,
    cfg_logging, cfg_bus,
    BusConfig,
)

CONFIG_PATH = str(pathlib.Path(__file__).parents[2] / "configs" / "config.yaml")


class TestLoadConfig:
    def test_loads_real_config(self):
        config = load_config(CONFIG_PATH, {})
        assert isinstance(config, dict)

    def test_all_required_sections_present(self):
        config = load_config(CONFIG_PATH, {})
        for section in ("platform", "airframe", "tracker", "guidance",
                        "watchdog", "mission", "logging"):
            assert section in config

    def test_missing_top_level_section_raises(self):
        yaml_text = "platform:\n  name: dev_pc\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            with pytest.raises(KeyError, match="airframe"):
                load_config(path, {})
        finally:
            os.unlink(path)

    def test_override_string_value(self):
        config = load_config(CONFIG_PATH, {"platform.name": "dev_pc"})
        assert config["platform"]["name"] == "dev_pc"

    def test_override_integer_value(self):
        # Width is int in yaml; override must coerce string "320" to int 320
        config = load_config(CONFIG_PATH, {"platform.camera.width": "320"})
        assert config["platform"]["camera"]["width"] == 320

    def test_override_bool_value(self):
        # bool("false") == True in Python — override must handle bool specially
        config = load_config(CONFIG_PATH, {"platform.realtime.control_sched_fifo": "false"})
        assert config["platform"]["realtime"]["control_sched_fifo"] is False

    def test_override_unknown_path_raises(self):
        with pytest.raises(KeyError):
            load_config(CONFIG_PATH, {"nonexistent.deep.key": "value"})


class TestAccessors:
    def setup_method(self):
        self.config = load_config(CONFIG_PATH, {})

    def test_cfg_platform(self):
        p = cfg_platform(self.config)
        assert p.name == "orange_pi5"
        assert p.camera.width == 640
        assert p.camera.fps == 60
        assert p.serial.baud == 115200
        assert p.inference.device == "rknn"
        assert p.realtime.kcf_cpu_core == 1
        assert p.realtime.control_sched_fifo is True

    def test_cfg_airframe(self):
        a = cfg_airframe(self.config)
        assert a.name == "flix_micro"
        assert a.mass_kg == pytest.approx(0.18)
        assert len(a.inertia) == 3
        assert a.control_limits.max_roll_deg == 35

    def test_cfg_tracker(self):
        t = cfg_tracker(self.config)
        assert t.kcf.detect_thresh == pytest.approx(0.5)
        assert t.nanotrack.exemplar_sz == 127
        assert t.fusion.confidence_gate == pytest.approx(0.7)

    def test_cfg_guidance(self):
        g = cfg_guidance(self.config)
        assert g.method == "pronav"
        assert g.pronav is not None
        assert g.pronav.N == pytest.approx(4.0)
        assert g.pronav.closing_vel_fallback == pytest.approx(2.0)

    def test_cfg_watchdog(self):
        w = cfg_watchdog(self.config)
        assert w.target_estimate_ms == 150
        assert w.fc_attitude_ms == 250
        assert w.fc_imu_ms == 50
        assert w.guidance_accel_ms == 100

    def test_cfg_mission_with_hil(self):
        m = cfg_mission(self.config)
        assert m.mode == "bench_hil"
        assert m.hil is not None
        assert m.hil.target_model == "constant_velocity"
        assert len(m.hil.initial_offset_m) == 3

    def test_cfg_mission_hil_none_when_absent(self):
        config = dict(self.config)
        config["mission"] = {"mode": "flight"}
        m = cfg_mission(config)
        assert m.hil is None

    def test_cfg_logging(self):
        lg = cfg_logging(self.config)
        assert lg.level == "INFO"
        assert lg.max_bytes == 10_485_760

    def test_cfg_bus_from_config(self):
        bus = cfg_bus(self.config)
        assert bus.ring_depth == 8

    def test_cfg_bus_defaults_when_section_absent(self):
        bus = cfg_bus({})
        assert bus == BusConfig(ring_depth=8)

    def test_cfg_tracker_ccv_field(self):
        config = load_config(CONFIG_PATH, {})
        tracker = cfg_tracker(config)
        assert tracker.ccv == "kcf"

    def test_cfg_tracker_ncv_field(self):
        config = load_config(CONFIG_PATH, {})
        tracker = cfg_tracker(config)
        assert tracker.ncv == "nanotrack"

    def test_cfg_tracker_mosse_is_mosse_config(self):
        from quadguide.core.config import MOSSEConfig
        config = load_config(CONFIG_PATH, {})
        tracker = cfg_tracker(config)
        assert isinstance(tracker.mosse, MOSSEConfig)

    def test_cfg_guidance_new_fields(self):
        g = cfg_guidance(self.config)
        assert g.throttle_hold == pytest.approx(0.55)
        assert g.pronav.closing_vel_ema_alpha == pytest.approx(0.3)
        assert g.pronav.closing_vel_min_area_rate == pytest.approx(0.001)
        assert g.pronav.closing_vel_area_scale == pytest.approx(5.0)

    def test_cfg_guidance_pure_pursuit_block(self):
        g = cfg_guidance(self.config)
        assert g.pure_pursuit is not None
        assert g.pure_pursuit.K == pytest.approx(6.0)
