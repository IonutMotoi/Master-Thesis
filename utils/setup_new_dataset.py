from pathlib import Path
import os
import cv2
import numpy as np
from pycocotools.mask import encode

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from utils.bbox_conversion import yolo_bboxes_to_pascal_voc


def extract_bboxes_from_masks(masks):
    boxes = np.zeros((masks.shape[2], 4), dtype=np.float32)
    x_any = np.any(masks, axis=0)
    y_any = np.any(masks, axis=1)
    for idx in range(masks.shape[2]):
        x = np.where(x_any[:, idx])[0]
        y = np.where(y_any[:, idx])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)

    return boxes


# Dataset
def get_new_dataset_dicts(root, source, pseudo_masks_path, naive=False):
    # Load the dataset subset defined by source
    assert source in ['train', 'validation', 'test'], \
        'source should be "train", "validation", "test"'

    if source == "train":
        source_path = Path(root, "train")
        pseudo_masks_path = Path(pseudo_masks_path)
    elif source == "validation":
        source_path = Path(root, "validation")
    else:  # source == "test":
        source_path = Path(root, "test")

    ids = [file.stem for file in source_path.glob("*.jpg")]

    dataset_dicts = []
    for img_id in ids:
        record = {}

        filename = str(source_path / f'{img_id}.jpg')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = img_id

        # Dimensions of the output of the model
        record["height"] = height
        record["width"] = width

        if source == "train":
            mask_path = pseudo_masks_path / f'{img_id}.npz'
            masks = np.load(mask_path)['arr_0'].astype(np.uint8)

            # Remove empty masks
            indices_to_remove = []
            for i in range(masks.shape[2]):
                if np.all((masks[:, :, i] == 0)):
                    indices_to_remove.append(i)
            if len(indices_to_remove) > 0:
                masks = np.delete(masks, indices_to_remove, axis=2)

        else:
            mask_path = source_path / f'{img_id}.npz'
            masks = np.load(mask_path)['arr_0'].astype(np.uint8)

        if source == "train" and not naive:
            box_path = source_path / f'{img_id}.txt'
            bboxes = np.loadtxt(box_path, delimiter=" ", dtype=np.float32)
            if bboxes.ndim == 2:
                bboxes = bboxes[:, 1:]
            else:  # only 1 instance
                bboxes = [bboxes[1:]]

            # Convert bboxes from YOLO format to Pascal VOC format
            bboxes = yolo_bboxes_to_pascal_voc(bboxes, img_height=height, img_width=width)
        else:
            bboxes = extract_bboxes_from_masks(masks)  # Pascal VOC format

        # Remove bboxes corresponding to empty masks
        if source == "train" and not naive:
            if len(indices_to_remove) > 0:
                bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in indices_to_remove]

        num_objs = masks.shape[2]
        assert (len(bboxes) == num_objs)

        objs = []
        for i in range(num_objs):
            obj = {
                "bbox": bboxes[i],
                "bbox_mode": BoxMode.XYXY_ABS,  # Pascal VOC bbox format
                "category_id": 0,
                "segmentation": encode(np.asarray(masks[:, :, i], order="F"))  # COCO’s compressed RLE format
            }
            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)
    return dataset_dicts


def setup_new_dataset(pseudo_masks_path=None, naive=False):
    data_path = "/thesis/new_dataset"

    for name in ["validation"]:  # ["train", "validation", "test"]:
        dataset_name = "new_dataset_" + name
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)

        DatasetCatalog.register(dataset_name, lambda d=name: get_new_dataset_dicts(data_path, d, pseudo_masks_path, naive=naive))
        MetadataCatalog.get(dataset_name).set(thing_classes=["grapes"])
