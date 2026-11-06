
# Updates notes

## 【2026/11/05】

Migration of `yolox-ne` from the original [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX):

* Python >= 3.10, PyTorch >= 2.0, Linux only; `pyproject.toml` packaging that does not import torch at build time.
* Removed Windows support, the MegEngine / ncnn / OpenVINO / nebullvm demos, TorchScript export, VOC support and `torch2trt`.
* Deployment is ONNX (opset 17 by default) and TensorRT 10.x: `tools/export_onnx.py`, then `tools/trt.py` to build an engine.
* Multi-GPU training always spawns processes; the RAM cache is a shared file in `/dev/shm`.
* Checkpoints load with `weights_only=True` (`YOLOX_TRUST_CKPT=1` opts out for trusted files).

Details and verification status: [SPEC.md](../SPEC.md).

## 【2021/08/19】

* Support image caching for faster training, which requires large system RAM. 
* Remove the dependence of apex and support torch amp training. 
* Optimize the preprocessing for faster training 
* Replace the older distort augmentation with new HSV aug for faster training and better performance. 

### 2X Faster training

The original project optimized the data preprocess and added image caching with the `--cache` flag:

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o [--cache]
                         yolox-m
                         yolox-l
                         yolox-x
```
* -d: number of gpu devices
* -b: total batch size, the recommended number for -b is num-gpu * 8
* --fp16: mixed precision training
* --cache: caching imgs into RAM to accelarate training, which need large system RAM.

### Higher performance

New models achieve **~1%** higher performance! See [Model_Zoo](model_zoo.md) for more details.

### Support torch amp

The original project added torch.cuda.amp training and dropped Apex.

### Breaking changes

The original project removed the normalization operation like -mean/std. This will make the old weights **incompatible**.

If you still want to use old weights, you can add `--legacy' in demo and eval:

```shell
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu] [--legacy]
```

and 

```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse] [--legacy]
                         yolox-m
                         yolox-l
                         yolox-x
```

But for the deployment demos, the original project no longer supports the old weights. Users could checkout to YOLOX version 0.1.0 to use legacy weights for deployment


