import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler

from yolox.data import DataLoader, YoloBatchSampler
from yolox.data.datasets.datasets_wrapper import Dataset


class _DS(Dataset):
    def __init__(self):
        super().__init__((32, 32))

    def __len__(self):
        return 8

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(index)), np.array([index])


@pytest.mark.parametrize("num_workers", [0, 2])
def test_yolox_dataloader_iterates_and_closes_mosaic(num_workers):
    ds = _DS()
    # same construction as Exp.get_data_loader
    bs = YoloBatchSampler(SequentialSampler(ds), batch_size=4, drop_last=False, mosaic=True)
    dl = DataLoader(ds, batch_sampler=bs, num_workers=num_workers)
    batches = list(dl)
    assert len(batches) == 2 and batches[0][0].shape == (4, 3, 4, 4)
    dl.close_mosaic()
    assert dl.batch_sampler.mosaic is False
