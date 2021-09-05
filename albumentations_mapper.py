import copy
from detectron2.data import detection_utils as utils
import albumentations as A
import numpy as np

class AlbumentationsMapper:
    def __init__(self, cfg, is_train=True):
        self.aug = A.Compose([
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    ])
        self.img_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT
        pass

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

        boxes = [ann['bbox'] for ann in dataset_dict['annotations']]
        labels = [ann['category_id'] for ann in dataset_dict['annotations']]
        masks = [ann['segmentation'] for ann in dataset_dict['annotations']]

        aug_img_annot = self.aug(image=image, bboxes=boxes, category_id=labels, masks=masks)

        image = aug_img_annot['image']
        h, w, _ = image.shape

        augm_boxes = np.array(aug_img_annot['bboxes'], dtype=np.float32)
        # sometimes bbox annotations go beyond image
        augm_boxes[:, :] = augm_boxes[:, :].clip(min=[0, 0, 0, 0], max=[w, h, w, h])
        augm_labels = np.array(aug_img_annot['category_id'])


        dataset_dict['annotations'] = [
            {
                'iscrowd': 0,
                'bbox': augm_boxes[i].tolist(),
                'category_id': augm_labels[i],
                'bbox_mode': BoxMode.XYWH_ABS,
            }
            for i in range(len(augm_boxes))
        ]