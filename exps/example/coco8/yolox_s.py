#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Smoke-test exp: YOLOX-S on the 8-image COCO8 sample converted with tools/yolo2coco.py
#   python tools/yolo2coco.py --src datasets/coco8 --out datasets/coco8_coco
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "datasets/coco8_coco"
        self.num_classes = 80

        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.data_num_workers = 2
        self.warmup_epochs = 1
        self.max_epoch = 40
        self.no_aug_epochs = 5
        self.eval_interval = 10
        self.print_interval = 1
        self.save_history_ckpt = False
