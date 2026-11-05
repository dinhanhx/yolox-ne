import pickle
import warnings

import numpy as np
import pytest
import torch

from yolox.utils.compat import grad_scaler, meshgrid, torch_load


def test_meshgrid_is_ij():
    a, b = torch.arange(2), torch.arange(3)
    got = meshgrid([a, b])
    want = torch.meshgrid(a, b, indexing="ij")
    assert all(torch.equal(g, w) for g, w in zip(got, want))


@pytest.mark.parametrize("ap", [
    0.31,  # what the trainer saves now
    pytest.param(np.float64(0.31), marks=pytest.mark.skipif(
        not hasattr(torch.serialization, "add_safe_globals"),
        reason="legacy checkpoints with numpy scalars need torch>=2.4 (add_safe_globals)")),
])
def test_torch_load_reads_trainer_style_checkpoint(tmp_path, ap):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    # older trainers stored np.float64 here: COCOeval.stats[0]
    state = {"start_epoch": 3, "model": model.state_dict(), "optimizer": opt.state_dict(),
             "best_ap": ap, "curr_ap": ap}
    path = tmp_path / "ckpt.pth"
    torch.save(state, path)
    ckpt = torch_load(path)
    assert float(ckpt["best_ap"]) == pytest.approx(0.31)
    assert ckpt["start_epoch"] == 3
    torch.nn.Linear(2, 2).load_state_dict(ckpt["model"])


class _Arbitrary:
    pass


def test_torch_load_refuses_arbitrary_objects_by_default(tmp_path):
    path = tmp_path / "evil.pth"
    torch.save({"x": _Arbitrary()}, path)
    with pytest.raises(pickle.UnpicklingError):
        torch_load(path)
    assert "x" in torch_load(path, weights_only=False)


def test_grad_scaler_disabled_has_no_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not grad_scaler(False).is_enabled()


@pytest.mark.skipif(not hasattr(torch.serialization, "get_safe_globals"), reason="needs torch>=2.5")
def test_numpy_scalar_allowed_under_both_numpy_module_names():
    # a checkpoint saved under numpy 1 names it numpy.core..., under numpy 2 numpy._core...
    torch_load.__globals__["_allow_numpy_scalars"]()
    allowed = {g[1] if isinstance(g, tuple) else f"{g.__module__}.{g.__qualname__}"
               for g in torch.serialization.get_safe_globals()}
    assert {"numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar"} <= allowed


def test_trust_env_var_loads_legacy_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "legacy.pth"
    torch.save({"x": _Arbitrary()}, path)
    monkeypatch.setenv("YOLOX_TRUST_CKPT", "1")
    assert "x" in torch_load(path)
