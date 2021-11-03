import copy
import logging
import numpy as np
import torch
from pycocotools.mask import encode
import cv2
import albumentations as A

import detectron2.data.transforms as T
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
    2. Use detectron tools to resize and crop the image (ResizeShortestEdge and RandomCrop)
    3. Apply augmentations/transforms to the image and annotations with Albumentations
    4. Prepare data and annotations and convert them to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train: bool = True, is_valid: bool = False):
        """
        Args:
            cfg: configuration
            is_train: whether it's used in training or inference
            is_valid: whether it's used for computing the validation loss
        """
        self.is_train = is_train
        self.is_valid = is_valid
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT
        self.resize_gen = detection_utils.build_transform_gen(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
        else:
            self.crop_gen = None

        # Define augmentations
        augmentations = get_augmentations(cfg, is_train)
        if is_train:
            self.augmentations = A.Compose(
                augmentations,
                bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels', 'bbox_ids']))
        else:
            self.augmentations = None

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
            # Convert H,W,C image to C,H,W tensor
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))

            if self.is_valid:
                image_shape = image.shape[1:]  # H, W
                annos = [anno for anno in dataset_dict.pop("annotations") if anno.get("iscrowd", 0) == 0]
                instances = detection_utils.annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format
                )
                dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
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
        transformed = self.augmentations(
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

        if self.crop_gen:
            # image crop using detectron2 tools
            crop_tfm = detection_utils.gen_crop_transform_with_instance(
                self.crop_gen.get_crop_size(image.shape[:2]),
                image.shape[:2],
                np.random.choice(dataset_dict["annotations"])
            )
            image = crop_tfm.apply_image(image)

        # image resize using detectron2 tools
        image, transforms = T.apply_transform_gens(self.resize_gen, image)
        if self.crop_gen:
            transforms = crop_tfm + transforms

        annos = [
            detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        dataset_dict['annotations'] = annos

        instances = detection_utils.annotations_to_instances(
            annos, image.shape[:2], mask_format=self.instance_mask_format
        )
        dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]
        # RGB to BGR required by the model
        image = image[:, :, ::-1]
        # Convert H,W,C image to C,H,W tensor
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))

        return dataset_dict


def pixel_dropout(image, p, **kwargs):
    """
    Set a fraction of pixels in images to zero.
    :param image:
    :param p: (list) a value ``p`` will be sampled from the interval ``[a, b]``
              per image and be used as the pixel's dropout probability.
    :param kwargs:
    :return: augmented image
    """
    assert isinstance(p, list), f"Expected p to be given as a list, got {type(p)}."
    assert len(p) == 2, (
            f"Expected p to be given as a list containing exactly 2 values, "
            f"got {len(p)} values.")
    assert p[0] <= p[1], (
            f"Expected p to be given as a list containing exactly 2 values "
            f"[a, b] with a <= b. Got {p[0]:.4f} and {p[1]:.4f}.")
    assert 0 <= p[0] <= 1.0 and 0 <= p[1] <= 1.0, (
            f"Expected p given as list to only contain values in the "
            f"interval [0.0, 1.0], got {p[0]:.4f} and {p[1]:.4f}.")

    height = image.shape[0]
    width = image.shape[1]
    # Dropout probability
    p = np.random.uniform(p[0], p[1])
    # Pixels to dropout
    dropouts = np.random.choice([0, 1], size=(height, width), p=[p, 1.0 - p]).astype('uint8')
    image = image * dropouts[:, :, np.newaxis]
    return image


def get_augmentations(cfg, is_train):
    augmentations = []

    if not is_train:
        return augmentations

    # Pixel Dropout
    if cfg.ALBUMENTATIONS.PIXEL_DROPOUT.ENABLED:
        augmentations.append(A.Lambda(
            name="pixel_dropout",
            image=lambda image, **kwargs: pixel_dropout(
                image,
                p=cfg.ALBUMENTATIONS.PIXEL_DROPOUT.DROPOUT_PROBABILITY),
            p=0.5))

    # Gaussian Noise
    if cfg.ALBUMENTATIONS.GAUSSIAN_NOISE.ENABLED:
        augmentations.append(A.GaussNoise())

    # Random Brightness Contrast
    if cfg.ALBUMENTATIONS.RANDOM_BRIGHTNESS_CONTRAST.ENABLED:
        augmentations.append(A.RandomBrightnessContrast())

    # Gaussian Blur
    if cfg.ALBUMENTATIONS.GAUSSIAN_BLUR.ENABLED:
        augmentations.append(A.GaussianBlur())

    return augmentations
