"""Guard: model manifests and board profiles must ship as package data.

EdgeCV was built by hatchling with a force-include for
edgecv/models/{manifests,profiles}; the merge translated that to setuptools
package-data. That translation fails SILENTLY in a source checkout, because
manifests resolve via Path(edgecv.__file__).parent, which works whether or
not the YAML was declared as package data. Only an installed wheel breaks,
and only at model-load time on the device.

These tests use importlib.resources, which reads through the package's
declared data rather than the filesystem, so they fail in an installed
environment where the YAML was not shipped.
"""
from importlib import resources

import pytest


def _names(subdir: str) -> set[str]:
    # Anchor on edgecv.models — a REAL package (it has __init__.py) — and reach
    # the data directory with joinpath. Do not use resources.files() directly on
    # "edgecv.models.manifests": that directory has no __init__.py, so it is at
    # best a namespace package and the call is not reliable across environments.
    return {p.name for p in resources.files("edgecv.models").joinpath(subdir).iterdir()}


def test_manifests_ship_as_package_data():
    names = _names("manifests")
    assert {"nanotrack.yaml", "siamfc_generic.yaml", "yolo11n.yaml"} <= names


def test_profiles_ship_as_package_data():
    names = _names("profiles")
    assert {"dev.yaml", "rk3588.yaml"} <= names


@pytest.mark.parametrize("name", ["nanotrack.yaml", "yolo11n.yaml"])
def test_manifest_is_readable_and_nonempty(name):
    text = (resources.files("edgecv.models")
            .joinpath("manifests", name)
            .read_text(encoding="utf-8"))
    assert "artifacts" in text
