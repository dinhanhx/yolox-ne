#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Convert a YOLO-format dataset (images/<split>, labels/<split>/*.txt) to the COCO layout YOLOX reads:

    <out>/annotations/instances_<split>2017.json
    <out>/<split>2017/<images>

YOLO label line: `class cx cy w h`, normalized to [0, 1]. COCO category ids are class index + 1.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def yolo_to_coco(src, out, names, splits=("train", "val")):
    src, out = Path(src), Path(out)
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    for split in splits:
        images, annotations = [], []
        (out / "annotations").mkdir(parents=True, exist_ok=True)
        (out / f"{split}2017").mkdir(parents=True, exist_ok=True)
        for img_path in sorted(p for p in (src / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS):
            h, w = cv2.imread(str(img_path)).shape[:2]
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
            shutil.copy2(img_path, out / f"{split}2017" / img_path.name)

            label_path = src / "labels" / split / f"{img_path.stem}.txt"
            lines = label_path.read_text().splitlines() if label_path.exists() else []
            for line in filter(str.strip, lines):
                cls, cx, cy, bw, bh = line.split()[:5]
                if not 0 <= int(cls) < len(names):
                    raise ValueError(f"class id {cls} out of range in {label_path} (have {len(names)} names)")
                bw, bh = float(bw) * w, float(bh) * h
                annotations.append({
                    "id": len(annotations) + 1, "image_id": img_id, "category_id": int(cls) + 1,
                    "bbox": [float(cx) * w - bw / 2, float(cy) * h - bh / 2, bw, bh],
                    "area": bw * bh, "iscrowd": 0,
                })
        with open(out / "annotations" / f"instances_{split}2017.json", "w") as f:
            json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
        print(f"{split}: {len(images)} images, {len(annotations)} boxes")


def main():
    parser = argparse.ArgumentParser("YOLO format -> COCO format")
    parser.add_argument("--src", required=True, help="YOLO dataset dir with images/ and labels/")
    parser.add_argument("--out", required=True, help="output COCO dataset dir")
    parser.add_argument("--names", default=None, help="file with one class name per line (default: 80 COCO)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    if args.names:
        names = [n.strip() for n in Path(args.names).read_text().splitlines() if n.strip()]
    else:
        from yolox.data.datasets import COCO_CLASSES as names
    yolo_to_coco(args.src, args.out, list(names), args.splits)


if __name__ == "__main__":
    main()
