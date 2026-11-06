from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yolox.core import Trainer
from yolox.utils import torch_load


def _fake_trainer(tmp_path):
    model = torch.nn.Linear(2, 2)
    return SimpleNamespace(
        rank=0, use_model_ema=False, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        epoch=3, best_ap=np.float64(0.25), file_name=str(tmp_path), args=SimpleNamespace(logger=None),
    )


@pytest.mark.parametrize("ap, want", [(None, None), (np.float64(0.3), 0.3)])
def test_save_ckpt_handles_missing_and_numpy_ap(tmp_path, ap, want):
    # the trainer calls save_ckpt(..., ap=None) after every epoch and with np.float64 after eval
    Trainer.save_ckpt(_fake_trainer(tmp_path), "latest", ap=ap)
    ckpt = torch_load(tmp_path / "latest_ckpt.pth")  # weights_only=True must accept it
    assert ckpt["curr_ap"] == want and ckpt["best_ap"] == pytest.approx(0.25) and ckpt["start_epoch"] == 4
