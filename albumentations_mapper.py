import copy
import logging
import numpy as np
import torch
from pycocotools.mask import encode

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import albumentations as A

"""
This file contains the mapping with Albumentations augmentation.
"""


class AlbumentationsMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    apply augmentations, and map it into a format used by the model.

    The callable does the following:
    1. Read the image from "file_name"
    2. Apply augmentations with Albumentations
    3. Applies cropping/geometric transforms to the image and annotations
    4. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train: bool = True):
        """
        Args:
            cfg: configuration
            is_train: whether it's used in training or inference
        """
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT

        # Log
        logger = logging.getLogger("detectron2")
        # mode = "training" if is_train else "inference"
        # logger.info(f"[AlbumentationsMapper] Augmentations used in {mode}: {self.augmentations}")
        if cfg.ALBUMENTATIONS.ENABLED:
            logger.info("############# ALBUMENTATIONS #################")
        if cfg.INPUT.PAD.ENABLED:
            logger.info(f"Padding images to size {cfg.INPUT.PAD.TARGET_WIDTH} "
                        f"x {cfg.INPUT.PAD.TARGET_HEIGHT} with value {cfg.INPUT.PAD.VALUE}")

        self.transform = A.Compose([
            A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels', 'bbox_ids']))

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        bboxes = [anno["bbox"] for anno in dataset_dict["annotations"]]
        bbox_mode = dataset_dict["annotations"][0]["bbox_mode"]
        masks = [anno["segmentation"] for anno in dataset_dict["annotations"]]
        class_labels = [anno["category_id"] for anno in dataset_dict["annotations"]]

        # Convert bboxes from the pascal_voc to the albumentations format
        bboxes = convert_pascal_voc_bboxes_to_albumentations(bboxes, height=image.shape[0], width=image.shape[1])

        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            masks=masks,
            class_labels=class_labels,
            bbox_ids=np.arange(len(bboxes))
        )
        image = transformed['image']
        bboxes = transformed['bboxes']
        masks = transformed['masks']
        class_labels = transformed['class_labels']
        bbox_ids = transformed['bbox_ids']

        # Filter the masks that don't have a corresponding bbox anymore
        # and convert uint8 masks of 0s and 1s into dicts in COCO’s compressed RLE format
        masks = [encode(np.asarray(masks[i], order="F")) for i in bbox_ids]

        # Convert bboxes from the albumentations format to the pascal_voc format
        bboxes = convert_albumentations_bboxes_to_pascal_voc(bboxes, height=image.shape[0], width=image.shape[1])

        assert len(bboxes) == len(class_labels), \
            "The number of bounding boxes should be equal to the number of class labels"
        assert len(bboxes) == len(masks), \
            "The number of bounding boxes should be equal to the number of masks"

        dataset_dict["annotations"] = [
            {
                "bbox": bboxes[i],
                "bbox_mode": bbox_mode,
                "segmentation": masks[i],
                "category_id": class_labels[i]
            }
            for i in range(len(bboxes))
        ]

        image_shape = image.shape[:2]  # h, w
        dataset_dict["height"] = image_shape[0]
        dataset_dict["width"] = image_shape[1]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annos = [anno for anno in dataset_dict.pop("annotations") if anno.get("iscrowd", 0) == 0]

        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


def convert_pascal_voc_bbox_to_albumentations(bbox, height, width):
    """
    Convert a bounding box from the pascal_voc format to the albumentations format:
    normalized coordinates of top-left and bottom-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.
    :param bbox: (tuple) Denormalized (pascal_voc format) bounding box `(x_min, y_min, x_max, y_max)`.
    :param height: (int) Image height.
    :param width: (int) Image width.
    :return: (tuple) Normalized (albumentations format) bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max = x_min / width, x_max / width
    y_min, y_max = y_min / height, y_max / height

    if -0.001 < x_min < 0.0:
        x_min = 0.0
    if -0.001 < y_min < 0.0:
        y_min = 0.0
    if 1 < x_max < 1.001:
        x_max = 1.0
    if 1 < y_max < 1.001:
        y_max = 1.0

    # Check that the bbox is in the range [0.0, 1.0]
    assert 0 <= x_min <= 1, "Expected bbox to be in the range [0.0, 1.0], got x_min = {x_min}.".format(x_min=x_min)
    assert 0 <= y_min <= 1, "Expected bbox to be in the range [0.0, 1.0], got y_min = {y_min}.".format(y_min=y_min)
    assert 0 <= x_max <= 1, "Expected bbox to be in the range [0.0, 1.0], got x_max = {x_max}.".format(x_max=x_max)
    assert 0 <= y_max <= 1, "Expected bbox to be in the range [0.0, 1.0], got y_max = {y_max}.".format(y_max=y_max)

    return x_min, y_min, x_max, y_max


def convert_albumentations_bbox_to_pascal_voc(bbox, height, width):
    """
    Convert a bounding box from the albumentations format to the pascal_voc format:
    denormalized coordinates of top-left and bottom-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)`.
    :param bbox: (tuple) Normalized (albumentations format) bounding box `(x_min, y_min, x_max, y_max)`.
    :param height: (int) Image height.
    :param width: (int) Image width.
    :return: (tuple) Denormalized (pascal_voc format) bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max = x_min * width, x_max * width
    y_min, y_max = y_min * height, y_max * height

    # Check that the bbox is in the range [0.0, 0.0, height, width]
    assert 0 <= x_min <= x_max, \
        "Expected x_min to be in the range [0.0, x_max], got x_min = {x_min}.".format(x_min=x_min)
    assert 0 <= y_min <= y_max, \
        "Expected y_min to be in the range [0.0, y_max], got y_min = {y_min}.".format(y_min=y_min)
    assert x_min <= x_max <= width, \
        "Expected x_max to be in the range [x_min, width], got x_max = {x_max}.".format(x_max=x_max)
    assert y_min <= y_max <= height, \
        "Expected y_max to be in the range [y_min, height], got y_max = {y_max}.".format(y_max=y_max)

    return x_min, y_min, x_max, y_max


def convert_pascal_voc_bboxes_to_albumentations(bboxes, height, width):
    """Convert a list of bounding boxes from the pascal_voc format to the albumentations format"""
    return [convert_pascal_voc_bbox_to_albumentations(bbox, height, width) for bbox in bboxes]


def convert_albumentations_bboxes_to_pascal_voc(bboxes, height, width):
    """Convert a list of bounding boxes from the albumentations format to the pascal_voc format"""
    return [convert_albumentations_bbox_to_pascal_voc(bbox, height, width) for bbox in bboxes]