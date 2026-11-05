#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.
"""TensorRT (>=10) helpers: build an engine from ONNX and run a serialized engine on torch tensors.

`tensorrt` is imported lazily: it is an optional extra, and JetPack ships its own build.
"""

import torch

__all__ = ["build_engine", "TRTModule"]


def _import_trt():
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "tensorrt>=10 is required (pip install 'yolox-ne[trt]', or use the JetPack build on Jetson)"
        ) from e
    return trt


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=4, max_batch=1):
    """Build a serialized TensorRT engine from an ONNX file.

    If the ONNX batch axis is dynamic, an optimization profile covering batch 1..max_batch is added.
    """
    trt = _import_trt()
    log = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(log)
    net = builder.create_network(0)  # TRT 10: explicit batch is the only mode
    parser = trt.OnnxParser(net, log)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)

    inp = net.get_input(0)
    if inp.shape[0] == -1:
        _, c, h, w = inp.shape
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, (1, c, h, w), (max_batch, c, h, w), (max_batch, c, h, w))
        cfg.add_optimization_profile(profile)

    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(blob)


class TRTModule:
    """Callable wrapper of a serialized engine: `module(images) -> output` (one tensor per output).

    Uses the TRT 10 tensor-name API (`execute_async_v3`). Inputs must be CUDA tensors.
    """

    def __init__(self, engine_path):
        trt = _import_trt()
        self._dtypes = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path} (built by a different TensorRT version?)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = self.engine.get_tensor_mode
        self.input_names = [n for n in names if mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in names if mode(n) == trt.TensorIOMode.OUTPUT]

    def __call__(self, *inputs):
        assert len(inputs) == len(self.input_names), f"expected {len(self.input_names)} inputs"
        keep = []  # keep tensors alive until the enqueued kernels finish
        for name, x in zip(self.input_names, inputs):
            x = x.to(self._dtypes[self.engine.get_tensor_dtype(name)]).contiguous()
            self.context.set_input_shape(name, tuple(x.shape))
            self.context.set_tensor_address(name, x.data_ptr())
            keep.append(x)
        outputs = []
        for name in self.output_names:
            out = torch.empty(
                tuple(self.context.get_tensor_shape(name)),
                dtype=self._dtypes[self.engine.get_tensor_dtype(name)],
                device=keep[0].device,
            )
            self.context.set_tensor_address(name, out.data_ptr())
            outputs.append(out)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
