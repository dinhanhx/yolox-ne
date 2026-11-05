#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse

from loguru import logger

from yolox.utils.trt import build_engine


def make_parser():
    parser = argparse.ArgumentParser("YOLOX TensorRT engine builder (ONNX -> engine)")
    parser.add_argument("--onnx", type=str, required=True, help="onnx file made by tools/export_onnx.py")
    parser.add_argument("--engine", type=str, default="model_trt.engine", help="output engine path")
    parser.add_argument("--no-fp16", action="store_true", help="build an FP32 engine")
    parser.add_argument("-w", "--workspace", type=int, default=4, help="workspace size in GB")
    parser.add_argument("-b", "--batch", type=int, default=1, help="max batch size (dynamic onnx only)")
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    build_engine(
        args.onnx, args.engine,
        fp16=not args.no_fp16, workspace_gb=args.workspace, max_batch=args.batch,
    )
    logger.info("saved TensorRT engine to {}".format(args.engine))


if __name__ == "__main__":
    main()
