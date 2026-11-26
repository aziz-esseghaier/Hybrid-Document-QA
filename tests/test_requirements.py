"""Tests that the pinned Kaggle versions agree with the package's dependency ranges."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_pins_satisfy_the_package_ranges():
    pins = {}
    for line in (ROOT / "requirements-kaggle.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            pins[canonicalize_name(name)] = version.strip()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    ranges = [Requirement(dep) for dep in project["dependencies"] + project["optional-dependencies"]["gpu"]]
    for req in ranges:
        name = canonicalize_name(req.name)
        assert name in pins, f"{req.name} is not pinned in requirements-kaggle.txt"
        assert req.specifier.contains(pins[name]), f"{req.name}=={pins[name]} is outside {req.specifier}"
