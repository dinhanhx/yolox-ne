#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import functools
import os

import numpy as np
import torch

__all__ = ["meshgrid", "torch_load", "grad_scaler"]


def meshgrid(*tensors):
    return torch.meshgrid(*tensors, indexing="ij")


@functools.cache
def _allow_numpy_scalars():
    # Trainer checkpoints store np.float64 (best_ap / curr_ap), which weights_only rejects.
    if not hasattr(torch.serialization, "add_safe_globals"):  # torch < 2.4
        return
    # a checkpoint names the unpickler callable np.core.* (saved under numpy 1) or np._core.* (numpy 2),
    # whichever numpy is installed now: allow both names
    scalar = np.float64(0).__reduce__()[0]
    allowed = [(scalar, "numpy.core.multiarray.scalar"), (scalar, "numpy._core.multiarray.scalar"), np.dtype]
    allowed += [getattr(np.dtypes, n) for n in ("Float32DType", "Float64DType") if hasattr(np, "dtypes")]
    torch.serialization.add_safe_globals(allowed)


def torch_load(path, map_location="cpu", weights_only=True):
    """torch.load that refuses arbitrary pickled objects unless weights_only=False is passed.

    Pass weights_only=False, or set YOLOX_TRUST_CKPT=1 for the CLI tools, only for trusted legacy
    checkpoints that fail to load (e.g. numpy scalars on torch < 2.4).
    """
    if os.environ.get("YOLOX_TRUST_CKPT") == "1":
        weights_only = False
    _allow_numpy_scalars()
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def grad_scaler(enabled: bool):
    # torch.amp.GradScaler appeared in torch 2.3; torch.cuda.amp.GradScaler is deprecated since 2.4
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
