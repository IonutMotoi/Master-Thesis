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
        self.augmentations = utils.build_augmentation(cfg, is_train)
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentations.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            self.recompute_boxes = cfg.MODEL.MASK_ON
        else:
            self.recompute_boxes = False

        if self.recompute_boxes:
            assert self.use_instance_mask, "recompute_boxes requires instance masks"

        # Log
        logger = logging.getLogger("detectron2")
        mode = "training" if is_train else "inference"
        logger.info(f"[AlbumentationsMapper] Augmentations used in {mode}: {self.augmentations}")
        if cfg.ALBUMENTATIONS.ENABLED:
            logger.info("############# ALBUMENTATIONS #################")
        if cfg.INPUT.PAD.ENABLED:
            logger.info(f"Padding images to size {cfg.INPUT.PAD.TARGET_WIDTH} "
                        f"x {cfg.INPUT.PAD.TARGET_HEIGHT} with value {cfg.INPUT.PAD.VALUE}")

        self.transform = A.Compose([
            A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

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
        masks = [anno["segmentation"] for anno in dataset_dict["annotations"]]
        class_labels = np.zeros(len(bboxes))

        transformed = self.transform(
            image=image,
            masks=masks,
            bboxes=bboxes,
            class_labels=class_labels
        )
        image = transformed['image']
        bboxes = transformed['bboxes']
        masks = transformed['masks']

        i = 0
        for anno in dataset_dict["annotations"]:
            anno["bbox"] = bboxes[i]
            anno["segmentation"] = masks[i]
            i += 1

        # aug_input = T.AugInput(image)
        # transforms = T.AugmentationList(self.augmentations)(aug_input)
        # image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if self.use_instance_mask:
                    # Convert uint8 mask of 0s and 1s into dict in COCO’s compressed RLE format
                    anno["segmentation"] = encode(np.asarray(anno["segmentation"], order="F"))
                else:
                    anno.pop("segmentation", None)

            # annos = [
            #     utils.transform_instance_annotations(obj, transforms, image_shape)
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]
            annos = [anno for anno in dataset_dict.pop("annotations") if anno.get("iscrowd", 0) == 0]

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
