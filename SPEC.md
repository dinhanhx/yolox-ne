# yolox-ne — Specification

Status as of 2026-11-05, branch `migrate/py310-torch2` (base `6ddff48`). Derived from `MIGRATION_PLAN.md` and the implemented code.
Labels: **[Verified]** = run in this session; **[Unverified]** = not run; **[Inference]** = reasoned, not run.

## 1. Targets

| Item | Requirement |
| --- | --- |
| Distribution / import name | `yolox-ne` / `yolox` |
| Python | `>=3.10` |
| PyTorch | `>=2.0` (`torchvision` matching) |
| numpy | any; floor job uses `numpy<2` |
| OS | Linux only |
| CUDA | `>=11` (GPU paths only) |
| TensorRT | `>=10,<11` (optional extra) |
| Deploy formats | ONNX, TensorRT engine |
| Dataset / metric | COCO only |

## 2. Packaging

- `pyproject.toml` is the only metadata source; build requires `setuptools>=77` and never imports torch. [Verified: wheel built and `pip install -e . --no-deps` done in an env without torch]
- `setup.py` is a shim that only maps `tools/` → `yolox.tools` and `exps/default/` → `yolox.exp.default`.
- Extras: `export` (`onnx>=1.13`, `onnxsim`, `onnxruntime`), `trt` (`tensorrt>=10,<11`), `test` (`pytest`, `tomli` on <3.11).
- `FastCOCOEvalOp` is not pre-compiled; it JIT-compiles on first use. [Verified: `g++` + torch 2.14]
- Removed: `requirements.txt`, `setup.cfg`, `ext_modules`.

## 3. Platform behavior

- `yolox.core.launch.launch` always uses `start_method="spawn"`; no `fork` branch, no Windows branch. [Verified: unit test]
- RAM image cache (`--cache ram`): `SharedImageCache` appends images to one file in `/dev/shm`; only `(path, offsets, shapes, dtypes)` is pickled. Reads attach by path and return a copy. The creating process deletes the file on GC/exit; copies in other processes never delete it. [Verified: roundtrip, pickle size < 10 kB, spawned-process read, owner-only cleanup]
- The cache is built in the launching process (`tools/train.py`) before `launch()`, so no cross-rank barrier exists. [Inference: host RAM ≈ 1× dataset; not measured]
- Limits: `/dev/shm` must fit the whole cache (Docker default 64 MB fails mid-cache with `OSError`); a SIGKILL leaves the file behind.
- `--cache disk` is unchanged.

## 4. Checkpoints and torch compat

- `yolox.utils.torch_load(path, map_location="cpu", weights_only=True)`: all repo `torch.load` calls go through it, except `torch.hub.load_state_dict_from_url` in `yolox/models/build.py`.
- On torch ≥ 2.4, numpy scalars are allow-listed under both `numpy.core.multiarray.scalar` and `numpy._core.multiarray.scalar`, since the trainer used to save `np.float64`. [Verified: torch 2.14 + numpy 1.26; the numpy-2 name is only checked as registered, not by loading a numpy-2 checkpoint]
- On torch < 2.4 checkpoints with numpy scalars are refused. `YOLOX_TRUST_CKPT=1` (or `weights_only=False`) loads trusted legacy files. [Verified]
- The trainer saves `best_ap` and `curr_ap` as `float`.
- `yolox.utils.grad_scaler(enabled)` uses `torch.amp.GradScaler("cuda", ...)` when present, else `torch.cuda.amp.GradScaler`. Autocast uses `torch.autocast("cuda", ...)`. `meshgrid` is `torch.meshgrid(..., indexing="ij")`.
- `yolox.data.DataLoader` (name-mangled `__initialized`) works on torch 2.14, with 0 and 2 workers. [Verified; torch 2.0.1 ran the same test with the other suites, passing]
- Known defect (pre-existing): `DataLoader` raises `UnboundLocalError` if neither `sampler` nor `batch_sampler` is passed.

## 5. ONNX export

- API: `tools.export_onnx.export_onnx(model, dummy_input, path, input_name="images", output_name="output", dynamic=False, opset=17)`.
- Uses public `torch.onnx.export`; passes `dynamo=False` when the kwarg exists (torch ≥ 2.5). The legacy TorchScript exporter is deprecated in torch 2.14. [Inference: it may be removed in a later torch]
- CLI: `python tools/export_onnx.py -n <name> -c <ckpt> [--dynamic] [-o 17] [--no-onnxsim] [--decode_in_inference]`; default opset 17; `SiLU` replaced via `replace_module`.
- Graph output is the raw head (`decode_in_inference=False`); NMS stays in Python (`yolox/utils/boxes.py`, torchvision `batched_nms`).
- Parity (FP32, ONNX Runtime CPU vs PyTorch): `atol=1e-4`, `rtol=1e-3`, static and dynamic batch, raw and decoded. [Verified: yolox-nano, random init, torch 2.14 and 2.0.1]
- Parity vs the Step 0 baseline: max abs diff 7.5e-8 (raw), 3.1e-5 (decoded). [Verified]
- Not measured: COCO mAP drop (plan target ≤ 0.1 FP32, ≤ 0.3 FP16). No dataset available. The mAP thresholds are plan proposals, [Inference].

## 6. TensorRT

- Build: `python tools/trt.py --onnx m.onnx --engine path.engine [--no-fp16] [-w GB] [-b MAXBATCH]`. Wraps `yolox.utils.trt.build_engine(onnx_path, engine_path, fp16=True, workspace_gb=4, max_batch=1)`. A dynamic batch axis gets one optimization profile, batch 1..`max_batch`.
- Run: `yolox.utils.trt.TRTModule(engine_path)(images_cuda) -> Tensor` (tuple if several outputs), via the TRT 10 tensor-name API (`execute_async_v3`) on torch's current stream. Input is cast to the engine's input dtype.
- `tools/demo.py --trt` and `tools/eval.py --trt` read `<output_dir>/<experiment-name>/model_trt.engine`. One torch forward pass runs first to set `head.hw` for the decoder.
- Optional import: importing `yolox` never requires `tensorrt`; `TRTModule`/`build_engine` raise `ImportError` if it is missing. Jetson uses JetPack's TRT.
- Verified on RTX 3060, CUDA 11.8, torch 2.7.1+cu118, TensorRT 10.13.0.35 (`tensorrt-cu11`): FP32 engine matches ONNX Runtime within the tolerance in §5 for static and dynamic batch; `tools/trt.py` → `tools/demo.py --trt` produced and saved detections on a random-init nano model. [Verified]
- Not verified: FP16 accuracy, `tools/eval.py --trt`, other GPUs, other TRT 10.x, CUDA 12 builds, Jetson. TRT 11 is excluded because `BuilderFlag.FP16` does not exist there. [Verified by introspection of 11.4]
- Known gaps: `--trt --fp16` warm-up uses FP32 input on a half model (pre-existing); `TRTModule` does not check that inputs are CUDA tensors; `tools/trt.py`'s default engine path is the current directory.

## 7. Removed scope

`demo/MegEngine`, `demo/ncnn`, `demo/OpenVINO`, `demo/nebullvm`, `tools/export_torchscript.py`, VOC dataset/evaluator/exps, `torch2trt`, Windows branches.
Kept: `demo/ONNXRuntime`, `demo/TensorRT`. The C++ TensorRT demo is still TRT 7 code and does not build against TRT 10 (not ported). [Inference from reading the source; not compiled]

## 8. Quality gates

- Lint: `ruff check yolox exps tools tests` (rules `E,F,W,A,PLC,PLE,PLW,I`, line length 120; ignores `PLC0415`, `PLW2901`, `A001`, `A002`; `F401/F403` allowed in `__init__.py`). `ruff format` is configured but not applied to the repo. [Verified: clean on ruff 0.17.3]
- Tests (`pytest tests`): packaging, scope (removals), launch/spawn, shared cache, dataloader, compat, ONNX parity, TRT parity (GPU-gated). Last results: torch 2.14 CPU 26 passed / 2 skipped; torch 2.0.1 CPU 24 passed / 4 skipped; CUDA env 26 passed (engine-build test run separately, passed). [Verified]
- CI (`.github/workflows/ci.yaml`): `lint`, `floor` (py3.10, torch 2.0.1, `constraints-floor.txt`), `latest` (py3.13), `gpu` (self-hosted, cu118 + cu128). [Unverified: never run on GitHub Actions]

## 9. Out of scope / future

- timm backbones (`TimmBackbone`, `exps/timm/`, `timm` extra, allow-list `cs3darknet_focus_{s,m,l,x}` and `darknet53`): specified in `MIGRATION_PLAN.md` "Future work"; not implemented.
- Multi-GPU spawn with `--cache ram`, CUDA 12 TRT builds, and the Read the Docs build are untested.
