from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_targets_py310_torch2():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert cfg["project"]["requires-python"] == ">=3.10"
    assert "torch>=2.0" in cfg["project"]["dependencies"]
    assert "torch" not in cfg["build-system"]["requires"]


def test_setup_py_does_not_import_torch_or_build_ext():
    src = (ROOT / "setup.py").read_text()
    assert "torch" not in src
    assert "ext_modules" not in src
