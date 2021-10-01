# import pycocotools
import os
import cv2
import numpy as np
from pycocotools.mask import encode

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog


# Dataset
def get_wgisd_dicts(root, source):
    # Load the dataset subset defined by source
    assert source in ['train', 'valid', 'test', 'augmented_valid', 'augmented_test'], \
        'source should be "train", "valid", "test", "augmented_valid" or "augmented_test"'

    if source == "train":
        source_path = os.path.join(root, 'train_split_masked.txt')
    elif source in ["valid", "augmented_valid"]:
        source_path = os.path.join(root, 'valid_split_masked.txt')
    else:  # source in ["test", "augmented_test"]
        source_path = os.path.join(root, 'test_masked.txt')

    if source == "augmented_valid":
        root = os.path.join(root, "augmented", "valid_masked")
    elif source == "augmented_test":
        root = os.path.join(root, "augmented", "test_masked")
    else:
        root = os.path.join(root, "data")

    with open(source_path, 'r') as fp:
        # Read all lines in file
        lines = fp.readlines()
        # Recover the items ids, removing the \n at the end
        ids = [l.rstrip() for l in lines]

    dataset_dicts = []
    for img_id in ids:
        record = {}

        filename = os.path.join(root, f'{img_id}.jpg')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = img_id
        record["height"] = height
        record["width"] = width

        box_path = os.path.join(root, f'{img_id}.txt')
        bboxes = np.loadtxt(box_path, delimiter=" ", dtype=np.float32)
        bboxes = bboxes[:, 1:]

        mask_path = os.path.join(root, f'{img_id}.npz')
        masks = np.load(mask_path)['arr_0'].astype(np.uint8)

        num_objs = masks.shape[2]
        assert (bboxes.shape[0] == num_objs)

        objs = []
        for i in range(num_objs):
            box = bboxes[i]
            mask = masks[:, :, i]

            # Boxes (x0, y0, w, h) in range [0, 1] (yolo format)
            # They are relative to the size of the image
            # Convert to (x0, y0, x1, y1) in absolute floating points coordinates (pascal_voc format)
            x1 = box[0] - box[2] / 2
            x2 = box[0] + box[2] / 2
            y1 = box[1] - box[3] / 2
            y2 = box[1] + box[3] / 2
            box = [x1 * width, y1 * height, x2 * width, y2 * height]

            if source in ["augmented_valid", "augmented_test"]:
                # Validation and test masks have to be encoded here for coco evaluator
                mask = encode(np.asarray(mask, order="F"))

            obj = {
                "bbox": box,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": mask,
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def setup_wgisd():
    data_path = "/thesis/wgisd"

    for d in ["train", "augmented_valid", "augmented_test"]:
        dataset_name = "wgisd_" + d
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)

        DatasetCatalog.register(dataset_name, lambda d=d: get_wgisd_dicts(data_path, d))
        MetadataCatalog.get(dataset_name).set(thing_classes=["grapes"])