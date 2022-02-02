from pathlib import Path
import os
import cv2
import numpy as np
from pycocotools.mask import encode

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from utils.bbox_conversion import yolo_bboxes_to_pascal_voc


# Dataset
def get_pseudo_bboxes_dicts(root, dataset_name, pseudo_masks_path):
    source_path = Path(root, dataset_name)
    pseudo_masks_path = Path(pseudo_masks_path, dataset_name)

    with open(os.path.join(root, "ids.txt"), 'r') as f:
        # Read all lines in file
        lines = f.readlines()
        # Recover the items ids, removing the \n at the end
        ids = [line.rstrip() for line in lines]

    dataset_dicts = []
    for img_id in ids:
        record = {}

        filename = str(source_path / f'{img_id}.png')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = img_id

        # Dimensions of the output of the model
        record["height"] = height
        record["width"] = width

        mask_path = pseudo_masks_path / f'{img_id}.npz'
        masks = np.load(mask_path)['arr_0'].astype(np.uint8)

        # Remove empty masks
        indices_to_remove = []
        for i in range(masks.shape[2]):
            if np.all((masks[:, :, i] == 0)):
                indices_to_remove.append(i)
        if len(indices_to_remove) > 0:
            masks = np.delete(masks, indices_to_remove, axis=2)

        box_path = source_path / f'{img_id}.txt'
        bboxes = np.loadtxt(box_path, delimiter=" ", dtype=np.float32)
        if bboxes.ndim == 2:
            bboxes = bboxes[:, 1:]
        else:  # only 1 instance
            bboxes = [bboxes[1:]]

        # Convert bboxes from YOLO format to Pascal VOC format
        bboxes = yolo_bboxes_to_pascal_voc(bboxes, img_height=height, img_width=width)

        # Remove bboxes corresponding to empty masks
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


def setup_pseudo_bboxes(pseudo_masks_path):
    data_path = "/thesis/videos"
    for dataset_name in ["closeup1", "closeup2", "video1", "video2", "video3"]:
        DatasetCatalog.register(dataset_name, lambda d=dataset_name: get_pseudo_bboxes_dicts(data_path, d, pseudo_masks_path))
        MetadataCatalog.get(dataset_name).set(thing_classes=["grapes"])
