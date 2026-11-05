#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import functools

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
    # the unpickler callable lives in np.core (numpy 1) or np._core (numpy 2); ask numpy itself
    allowed = [np.float64(0).__reduce__()[0], np.dtype]
    allowed += [getattr(np.dtypes, n) for n in ("Float32DType", "Float64DType") if hasattr(np, "dtypes")]
    torch.serialization.add_safe_globals(allowed)


def torch_load(path, map_location="cpu", weights_only=True):
    """torch.load that refuses arbitrary pickled objects unless weights_only=False is passed.

    Pass weights_only=False only for trusted legacy checkpoints that fail to load.
    """
    _allow_numpy_scalars()
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def grad_scaler(enabled: bool):
    # torch.amp.GradScaler appeared in torch 2.3; torch.cuda.amp.GradScaler is deprecated since 2.4
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
