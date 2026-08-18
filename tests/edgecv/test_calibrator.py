"""Tests for ScoreCalibrator implementations (spec §4.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from edgecv.fusion.calibrator import LinearCalibrator, SigmoidCalibrator


class TestLinearCalibrator:
    def test_default_params(self):
        cal = LinearCalibrator()
        assert cal.low == 3.0
        assert cal.high == 15.0
        assert cal.params == {"low": 3.0, "high": 15.0}

    def test_below_low_clamps_to_zero(self):
        cal = LinearCalibrator(low=3.0, high=15.0)
        assert cal.calibrate(2.0) == 0.0
        assert cal.calibrate(0.0) == 0.0
        assert cal.calibrate(-10.0) == 0.0

    def test_above_high_clamps_to_one(self):
        cal = LinearCalibrator(low=3.0, high=15.0)
        assert cal.calibrate(15.0) == 1.0
        assert cal.calibrate(20.0) == 1.0
        assert cal.calibrate(100.0) == 1.0

    def test_midpoint(self):
        cal = LinearCalibrator(low=3.0, high=15.0)
        # At midpoint (3+15)/2 = 9, value should be 0.5
        result = cal.calibrate(9.0)
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_quarter_point(self):
        cal = LinearCalibrator(low=0.0, high=100.0)
        assert cal.calibrate(25.0) == pytest.approx(0.25, abs=1e-9)
        assert cal.calibrate(75.0) == pytest.approx(0.75, abs=1e-9)

    def test_zero_range_does_not_divide_by_zero(self):
        cal = LinearCalibrator(low=5.0, high=5.0)
        # Range is zero; denominator becomes max(0, 1e-9) = 1e-9
        result = cal.calibrate(5.0)
        assert result == 0.0  # (5-5)/1e-9 clamped
        result = cal.calibrate(10.0)
        assert result == 1.0  # clamped

    def test_negative_range(self):
        cal = LinearCalibrator(low=10.0, high=0.0)
        # What happens when low > high? The clamp handles it, but results may not
        # be meaningful. At least it doesn't crash.
        result = cal.calibrate(5.0)
        assert 0.0 <= result <= 1.0

    def test_isinstance_score_calibrator(self):
        from edgecv.fusion.calibrator import ScoreCalibrator
        assert isinstance(LinearCalibrator(), ScoreCalibrator)


class TestSigmoidCalibrator:
    def test_default_params(self):
        cal = SigmoidCalibrator()
        assert cal.centre == 0.5
        assert cal.steepness == 10.0
        assert cal.params == {"centre": 0.5, "steepness": 10.0}

    def test_at_centre_returns_05(self):
        cal = SigmoidCalibrator(centre=0.5, steepness=10.0)
        result = cal.calibrate(0.5)
        assert result == pytest.approx(0.5, abs=1e-3)

    def test_above_centre_above_05(self):
        cal = SigmoidCalibrator(centre=0.5, steepness=10.0)
        result = cal.calibrate(0.7)
        assert result > 0.5

    def test_below_centre_below_05(self):
        cal = SigmoidCalibrator(centre=0.5, steepness=10.0)
        result = cal.calibrate(0.3)
        assert result < 0.5

    def test_very_high_raw_approaches_one(self):
        cal = SigmoidCalibrator(centre=0.5, steepness=10.0)
        result = cal.calibrate(1.0)
        # sigmoid(10*0.5)=sigmoid(5)=0.9933; relax tolerance
        assert result == pytest.approx(1.0, abs=1e-2)

    def test_very_low_raw_approaches_zero(self):
        cal = SigmoidCalibrator(centre=0.5, steepness=10.0)
        result = cal.calibrate(0.0)
        # sigmoid(10*(-0.5))=sigmoid(-5)=0.00669; relax tolerance
        assert result == pytest.approx(0.0, abs=1e-2)

    def test_steepness_controls_sharpness(self):
        # Higher steepness = sharper transition
        gentle = SigmoidCalibrator(centre=0.5, steepness=2.0)
        sharp = SigmoidCalibrator(centre=0.5, steepness=20.0)
        # At 0.6, gentle should be closer to 0.5, sharp closer to 1.0
        g_result = gentle.calibrate(0.6)
        s_result = sharp.calibrate(0.6)
        assert s_result > g_result

    def test_custom_centre(self):
        cal = SigmoidCalibrator(centre=0.3, steepness=10.0)
        result = cal.calibrate(0.3)
        assert result == pytest.approx(0.5, abs=1e-3)

    def test_output_range(self):
        cal = SigmoidCalibrator(centre=0.5, steepness=5.0)
        for raw in np.linspace(-2.0, 2.0, 50):
            result = cal.calibrate(float(raw))
            assert 0.0 < result < 1.0, f"raw={raw} -> result={result}"

    def test_isinstance_score_calibrator(self):
        from edgecv.fusion.calibrator import ScoreCalibrator
        assert isinstance(SigmoidCalibrator(), ScoreCalibrator)
