import importlib.util
import sys

import numpy as np
import pytest
import torch

HAS_TRT = importlib.util.find_spec("tensorrt") is not None


def test_module_imports_without_tensorrt():
    # tensorrt is an optional extra (JetPack provides it): importing yolox must never require it
    import yolox.utils.trt as trt_utils

    assert hasattr(trt_utils, "build_engine") and hasattr(trt_utils, "TRTModule")
    if not HAS_TRT:
        assert "tensorrt" not in sys.modules
        with pytest.raises(ImportError, match="tensorrt"):
            trt_utils.TRTModule("missing.engine")


def test_no_torch2trt_left():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    hits = [str(p) for d in ("yolox", "tools") for p in (root / d).rglob("*.py") if "torch2trt" in p.read_text()]
    assert hits == []


@pytest.mark.skipif(not (HAS_TRT and torch.cuda.is_available()), reason="needs tensorrt + CUDA GPU")
@pytest.mark.parametrize("dynamic", [False, True])
def test_trt_engine_matches_onnxruntime(tmp_path, dynamic):
    import onnxruntime as ort
    from torch import nn

    from tools.export_onnx import export_onnx
    from yolox.exp import get_exp
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module
    from yolox.utils.trt import TRTModule, build_engine

    torch.manual_seed(0)
    model = replace_module(get_exp(exp_name="yolox-nano").get_model().eval(), nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    x = torch.rand(2 if dynamic else 1, 3, 416, 416) * 255
    onnx_path, engine_path = str(tmp_path / "m.onnx"), str(tmp_path / "m.engine")
    export_onnx(model, x[:1], onnx_path, dynamic=dynamic, opset=17)
    build_engine(onnx_path, engine_path, fp16=False, workspace_gb=1, max_batch=2 if dynamic else 1)

    want = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(
        None, {"images": x.numpy()[: (2 if dynamic else 1)]})[0]
    got = TRTModule(engine_path)(x.cuda()).cpu().numpy()
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-3)  # FP32 tolerance from MIGRATION_PLAN.md
