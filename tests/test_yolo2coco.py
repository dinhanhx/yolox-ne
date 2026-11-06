import cv2
import numpy as np
import pytest
from pycocotools.coco import COCO

from tools.yolo2coco import yolo_to_coco

NAMES = ["a", "b", "c"]


def _make(root, split, name, label_lines, w=100, h=200):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), np.zeros((h, w, 3), np.uint8))
    if label_lines is not None:
        (root / "labels" / split / f"{name}.txt").write_text("\n".join(label_lines))


def test_converts_boxes_to_absolute_xywh(tmp_path):
    src, out = tmp_path / "yolo", tmp_path / "coco"
    _make(src, "train", "img1", ["1 0.5 0.5 0.5 0.25"])  # image is 100 wide, 200 high
    yolo_to_coco(src, out, NAMES, splits=("train",))
    coco = COCO(str(out / "annotations" / "instances_train2017.json"))
    (img,) = coco.dataset["images"]
    assert (img["width"], img["height"], img["file_name"]) == (100, 200, "img1.jpg")
    assert (out / "train2017" / "img1.jpg").exists()
    (ann,) = coco.dataset["annotations"]
    assert ann["bbox"] == pytest.approx([25, 75, 50, 50]) and ann["area"] == pytest.approx(2500)
    assert ann["category_id"] == 2 and ann["iscrowd"] == 0 and ann["image_id"] == img["id"]
    assert [c["name"] for c in coco.dataset["categories"]] == NAMES


def test_image_without_label_file_has_no_annotations(tmp_path):
    src, out = tmp_path / "yolo", tmp_path / "coco"
    _make(src, "val", "img1", None)
    yolo_to_coco(src, out, NAMES, splits=("val",))
    coco = COCO(str(out / "annotations" / "instances_val2017.json"))
    assert len(coco.dataset["images"]) == 1 and coco.dataset["annotations"] == []


def test_class_id_outside_names_is_rejected(tmp_path):
    src, out = tmp_path / "yolo", tmp_path / "coco"
    _make(src, "train", "img1", ["7 0.5 0.5 0.1 0.1"])
    with pytest.raises(ValueError, match="class id 7"):
        yolo_to_coco(src, out, NAMES, splits=("train",))
