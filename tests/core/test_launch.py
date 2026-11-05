import importlib
from argparse import Namespace

# `yolox.core.launch` the module is shadowed by the `launch` function in yolox.core
launch_mod = importlib.import_module("yolox.core.launch")


def test_launch_always_spawns_even_with_cache(monkeypatch):
    seen = {}
    monkeypatch.setattr(launch_mod.mp, "start_processes", lambda *a, **k: seen.update(k))
    launch_mod.launch(lambda *a: None, 2, args=(None, Namespace(cache="ram")), dist_url="tcp://127.0.0.1:1")
    assert seen["start_method"] == "spawn"
