import configparser
from pathlib import Path

UNIT = Path(__file__).resolve().parents[2] / "systemd" / "quadguide.service"


def _parse():
    cp = configparser.ConfigParser()
    cp.optionxform = str  # preserve directive case: ExecStart, not execstart
    cp.read(UNIT)
    return cp


def test_unit_sections_and_directives():
    cp = _parse()
    assert {"Unit", "Service", "Install"} <= set(cp.sections())

    svc = cp["Service"]
    assert svc["Type"] == "simple"
    assert svc["User"] == "root"
    assert "scripts/run.py" in svc["ExecStart"]
    assert svc["Restart"] == "always"
    assert svc["RestartSec"] == "3"
    assert svc["KillMode"] == "mixed"
    assert svc["TimeoutStopSec"] == "10"
    assert svc["LimitRTPRIO"] == "99"

    assert cp["Unit"]["After"] == "network.target"
    assert cp["Install"]["WantedBy"] == "multi-user.target"


def test_unit_keeps_installer_tokens():
    text = UNIT.read_text()
    for tok in ("@REPO_DIR@", "@PYTHON@", "@CONFIG@", "@LOG_DIR@"):
        assert tok in text, f"missing installer token {tok}"
