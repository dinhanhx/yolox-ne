import numpy as np
import pytest
import torch
from torch import nn

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from tools.export_onnx import export_onnx  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.models.network_blocks import SiLU  # noqa: E402
from yolox.utils import replace_module  # noqa: E402

ATOL, RTOL = 1e-4, 1e-3  # FP32 parity tolerance agreed in MIGRATION_PLAN.md


def _model(decode):
    torch.manual_seed(0)
    model = get_exp(exp_name="yolox-nano").get_model().eval()
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = decode
    return model


def _run(path, x):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"images": x.numpy()})[0]


@pytest.mark.parametrize("decode", [False, True])
def test_onnx_matches_pytorch(tmp_path, decode):
    model = _model(decode)
    x = torch.rand(1, 3, 416, 416) * 255
    out = tmp_path / "m.onnx"
    export_onnx(model, x, str(out), dynamic=False, opset=17)
    with torch.no_grad():
        want = model(x).numpy()
    np.testing.assert_allclose(_run(out, x), want, atol=ATOL, rtol=RTOL)


def test_onnx_dynamic_batch(tmp_path):
    model = _model(False)
    out = tmp_path / "m.onnx"
    export_onnx(model, torch.rand(1, 3, 416, 416), str(out), dynamic=True, opset=17)
    x = torch.rand(2, 3, 416, 416) * 255
    with torch.no_grad():
        want = model(x).numpy()
    got = _run(out, x)
    assert got.shape[0] == 2
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)
