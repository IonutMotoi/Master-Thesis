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
    assert source in ['train', 'valid', 'test', 'augmented_valid', 'augmented_test', 'augmented_test_detection'], \
        'source should be "train", "valid", "test", "augmented_valid", "augmented_test" or "augmented_test_detection"'

    has_masks = True

    if source == "train":
        source_path = os.path.join(root, 'train_split_masked.txt')
    elif source in ["valid", "augmented_valid"]:
        source_path = os.path.join(root, 'valid_split_masked.txt')
    elif source in ["test", "augmented_test"]:
        source_path = os.path.join(root, 'test_masked.txt')
    else:  # source == "augmented_test_detection"
        source_path = os.path.join(root, 'test.txt')

    if source == "augmented_valid":
        root = os.path.join(root, "augmented", "valid_masked")
    elif source == "augmented_test":
        root = os.path.join(root, "augmented", "test_masked")
    elif source == "augmented_test_detection":
        root = os.path.join(root, "augmented", "test_detection")
        has_masks = False
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

        num_objs = bboxes.shape[0]

        if has_masks:
            mask_path = os.path.join(root, f'{img_id}.npz')
            masks = np.load(mask_path)['arr_0'].astype(np.uint8)
            assert (masks.shape[2] == num_objs)

        objs = []
        for i in range(num_objs):
            box = bboxes[i]
            # Boxes (x0, y0, w, h) in range [0, 1] (yolo format)
            # They are relative to the size of the image
            # Convert to (x0, y0, x1, y1) in absolute floating points coordinates (pascal_voc format)
            x1 = box[0] - box[2] / 2
            x2 = box[0] + box[2] / 2
            y1 = box[1] - box[3] / 2
            y2 = box[1] + box[3] / 2
            box = [x1 * width, y1 * height, x2 * width, y2 * height]

            obj = {
                "bbox": box,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            if has_masks:
                obj["segmentation"] = encode(np.asarray(masks[:, :, i], order="F"))

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def setup_wgisd():
    data_path = "/thesis/wgisd"

    for d in ["train", "augmented_valid", "augmented_test", "augmented_test_detection"]:
        dataset_name = "wgisd_" + d
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)

        DatasetCatalog.register(dataset_name, lambda d=d: get_wgisd_dicts(data_path, d))
        MetadataCatalog.get(dataset_name).set(thing_classes=["grapes"])