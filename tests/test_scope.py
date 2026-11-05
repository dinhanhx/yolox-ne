from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_only_onnx_and_trt_demos_remain():
    assert sorted(p.name for p in (ROOT / "demo").iterdir()) == ["ONNXRuntime", "TensorRT"]
    assert not (ROOT / "tools" / "export_torchscript.py").exists()


def test_voc_support_removed():
    import yolox.data.datasets as datasets
    import yolox.evaluators as evaluators

    assert not hasattr(datasets, "VOCDetection")
    assert not hasattr(evaluators, "VOCEvaluator")
    assert not (ROOT / "exps" / "example" / "yolox_voc").exists()
