import copy
import logging
import numpy as np
import torch
from pycocotools.mask import encode
import cv2
import albumentations as A
from detectron2.data import detection_utils

from utils.bbox_conversion import pascal_voc_bboxes_to_albumentations, albumentations_bboxes_to_pascal_voc

"""
This file contains the mapping with Albumentations augmentation.
"""


class AlbumentationsMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    apply augmentations, and map it into a format used by the model.

    The callable does the following:
    1. Read the image from "file_name"
    2. Apply augmentations/transforms to the image and annotations with Albumentations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train: bool = True):
        """
        Args:
            cfg: configuration
            is_train: whether it's used in training or inference
        """
        self.is_train = is_train
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT

        # Define augmentations
        augmentations = get_augmentations(cfg, is_train)
        if is_train:
            self.transform = A.Compose(
                augmentations,
                bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels', 'bbox_ids']))
        else:
            self.transform = None

        # Log
        logger = logging.getLogger("detectron2")
        mode = "training" if is_train else "inference"
        logger.info("############# ALBUMENTATIONS #################")
        logger.info(f"[AlbumentationsMapper] Augmentations used in {mode}:")
        for aug in augmentations:
            logger.info(aug)
        logger.info("##############################################")

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        # Evaluation
        if not self.is_train:
            image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
            detection_utils.check_image_size(dataset_dict, image)
            dataset_dict.pop("annotations", None)
            # Convert H,W,C image to C,H,W tensor
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))
            return dataset_dict

        # Training
        image = detection_utils.read_image(dataset_dict["file_name"], format="RGB")  # RGB required by albumentations
        detection_utils.check_image_size(dataset_dict, image)

        bboxes = [anno["bbox"] for anno in dataset_dict["annotations"]]
        bbox_mode = dataset_dict["annotations"][0]["bbox_mode"]
        masks = [anno["segmentation"] for anno in dataset_dict["annotations"]]
        class_labels = [anno["category_id"] for anno in dataset_dict["annotations"]]

        # Convert bboxes from the pascal_voc to the albumentations format
        bboxes = pascal_voc_bboxes_to_albumentations(bboxes, height=image.shape[0], width=image.shape[1])

        # Apply transformations
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            masks=masks,
            class_labels=class_labels,
            bbox_ids=np.arange(len(bboxes))
        )
        image = transformed["image"]
        bboxes = transformed["bboxes"]
        masks = transformed["masks"]
        class_labels = transformed["class_labels"]
        bbox_ids = transformed["bbox_ids"]

        # Filter the masks that don't have a corresponding bbox anymore
        # and convert uint8 masks of 0s and 1s into dicts in COCO’s compressed RLE format
        masks = [encode(np.asarray(masks[i], order="F")) for i in bbox_ids]

        # Convert bboxes from the albumentations format to the pascal_voc format
        bboxes = albumentations_bboxes_to_pascal_voc(bboxes, height=image.shape[0], width=image.shape[1])

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

        image_shape = image.shape[:2]  # H, W
        dataset_dict["height"] = image_shape[0]
        dataset_dict["width"] = image_shape[1]
        # RGB to BGR required by the model
        image = image[:, :, ::-1]
        # Convert H,W,C image to C,H,W tensor
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))

        annos = [anno for anno in dataset_dict.pop("annotations") if anno.get("iscrowd", 0) == 0]

        instances = detection_utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        return dataset_dict


def pixel_dropout(image, p, **kwargs):
    print(p)
    height = image.shape[0]
    width = image.shape[1]
    print(height)
    print(width)
    return image


def get_augmentations(cfg, is_train):
    augmentations = []

    # Longest Max Size
    if cfg.ALBUMENTATIONS.LONGEST_MAX_SIZE.ENABLED:
        augmentations.append(A.LongestMaxSize(max_size=cfg.ALBUMENTATIONS.LONGEST_MAX_SIZE.VALUE))
    # Pad
    if cfg.ALBUMENTATIONS.PAD.ENABLED:
        augmentations.append(A.PadIfNeeded(
            min_height=cfg.ALBUMENTATIONS.PAD.TARGET_HEIGHT,
            min_width=cfg.ALBUMENTATIONS.PAD.TARGET_WIDTH,
            border_mode=cv2.BORDER_CONSTANT,
            value=cfg.ALBUMENTATIONS.PAD.VALUE,
            mask_value=cfg.ALBUMENTATIONS.PAD.MASK_VALUE
        ))

    # Only Longest Max Size and Pad during evaluation
    # Those transformations were already applied, here are returned for print purposes
    if not is_train:
        return augmentations

    # Horizontal Flip
    if cfg.ALBUMENTATIONS.HORIZONTAL_FLIP.ENABLED:
        augmentations.append(A.HorizontalFlip())

    # Gaussian Blur
    if cfg.ALBUMENTATIONS.GAUSSIAN_BLUR.ENABLED:
        augmentations.append(A.GaussianBlur())

    # Gaussian Noise
    if cfg.ALBUMENTATIONS.GAUSSIAN_NOISE.ENABLED:
        augmentations.append(A.GaussNoise())

    # Random Brightness Contrast
    if cfg.ALBUMENTATIONS.RANDOM_BRIGHTNESS_CONTRAST.ENABLED:
        augmentations.append(A.RandomBrightnessContrast())

    # Pixel Dropout
    if cfg.ALBUMENTATIONS.PIXEL_DROPOUT.ENABLED:
        augmentations.append(A.Lambda(
            name="pixel_dropout",
            image=lambda image, **kwargs: pixel_dropout(image, p=cfg.ALBUMENTATIONS.PIXEL_DROPOUT.DROP_PROBABILITY),
            p=0.5))

    return augmentations
