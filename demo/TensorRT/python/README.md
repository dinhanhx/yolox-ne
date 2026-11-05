# YOLOX-TensorRT in Python

Engines are built from ONNX with TensorRT 10.x (`tensorrt>=10,<11`). On Jetson use the TensorRT that ships with JetPack instead of pip.

## Install

```shell
pip install -e ".[export,trt]"
```

## Convert model

1. Export ONNX (raw head output, no decode; add `--dynamic` for a dynamic batch axis):
   ```shell
   python tools/export_onnx.py -n yolox-s -c your_ckpt.pth --output-name yolox_s.onnx
   ```
2. Build the engine (FP16 by default; `--no-fp16` for FP32, `-b` is the max batch of a `--dynamic` ONNX):
   ```shell
   python tools/trt.py --onnx yolox_s.onnx --engine YOLOX_outputs/yolox_s/model_trt.engine
   ```

`tools/demo.py` and `tools/eval.py` read `<output_dir>/<experiment-name>/model_trt.engine`. Engines are tied to the GPU and TensorRT version that built them.

## Demo

```shell
python tools/demo.py image -n yolox-s --trt --save_result
```
or
```shell
python tools/demo.py image -f exps/default/yolox_s.py --trt --save_result
```
