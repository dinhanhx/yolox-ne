import gc
import multiprocessing as mp
import os
import pickle

import numpy as np

from yolox.data.datasets.datasets_wrapper import CacheDataset, cache_read_img

N = 6


def _img(i):
    rng = np.random.default_rng(i)
    return rng.integers(0, 255, size=(40 + 10 * i, 50 + 5 * i, 3), dtype=np.uint8)


class _DS(CacheDataset):
    def __init__(self):
        super().__init__((416, 416), num_imgs=N, cache=True, cache_type="ram")

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return _img(index)

    def __len__(self):
        return N

    def pull_item(self, index):
        return self.read_img(index)


def _child_read(ds, q):
    q.put([ds.read_img(i).tobytes() == _img(i).tobytes() for i in range(N)])


def test_ram_cache_roundtrip_variable_shapes():
    ds = _DS()
    for i in range(N):
        assert np.array_equal(ds.read_img(i), _img(i))


def test_ram_cache_pickle_is_small_and_readable():
    ds = _DS()
    blob = pickle.dumps(ds)
    assert len(blob) < 10_000  # images must not travel through pickle
    ds2 = pickle.loads(blob)
    assert all(np.array_equal(ds2.read_img(i), _img(i)) for i in range(N))


def test_ram_cache_readable_from_spawned_process():
    ds = _DS()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child_read, args=(ds, q))
    p.start()
    ok = q.get(timeout=60)
    p.join()
    assert all(ok)


def test_cache_file_removed_by_owner_only():
    ds = _DS()
    path = ds.imgs.path
    assert os.path.exists(path)
    ds2 = pickle.loads(pickle.dumps(ds))
    del ds2
    gc.collect()
    assert os.path.exists(path), "a non-owner copy must not delete the shared file"
    del ds
    gc.collect()
    assert not os.path.exists(path)
