# YOLOX-TensorRT in C++

> **Not ported yet:** this C++ demo is written for TensorRT 7 (`getBindingIndex`, `enqueue`, `destroy`, tensor names `input_0`/`output_0`) and does not build against TensorRT 10, which the Python tooling now requires. Use the Python demo until it is ported.

Engines are built from ONNX by `tools/trt.py`; this C++ demo only loads a serialized engine and does not include model converting or constructing.


## Step 1: Prepare serialized engine file

Follow the trt [python demo README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/TensorRT/python/README.md) to export ONNX and build the serialized engine file, then copy it to this demo dir as `model_trt.engine`.


## Step 2: build the demo

Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT.

And you should set the TensorRT path and CUDA path in CMakeLists.txt.

If you train your custom dataset, you may need to modify the value of `num_class`.

```c++
const int num_class = 80;
```

Install opencv with ```sudo apt-get install libopencv-dev``` (we don't need a higher version of opencv like v3.3+). 

build the demo:

```shell
mkdir build
cd build
cmake ..
make
```

Then run the demo:

```shell
./yolox ../model_trt.engine -i ../../../../assets/dog.jpg
```

or

```shell
./yolox <path/to/your/engine_file> -i <path/to/image>
```

NOTE: for `trtexec` users, modify `INPUT_BLOB_NAME` and `OUTPUT_BLOB_NAME` as the following code.
```
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output";
```

Here is the command to convert the small onnx model to tensorrt engine file:
```
trtexec --onnx=yolox_s.onnx --saveEngine=yolox_s.trt
```
