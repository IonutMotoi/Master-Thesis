import os.path

import numpy as np
import torch
import tqdm
from detectron2.data.detection_utils import read_image

from pseudo_labeling.predictor import MasksFromBboxesPredictor
from utils.bbox_conversion import yolo_bboxes_to_pascal_voc
from utils.save import save_masks


class MasksFromBboxes:
    def __init__(self, cfg, ids_txt, data_folder, dest_folder):
        with open(ids_txt, 'r') as f:
            # Read all lines in file
            lines = f.readlines()
            # Recover the items ids, removing the \n at the end
            self.ids = [line.rstrip() for line in lines]

        self.predictor = MasksFromBboxesPredictor(cfg)
        self.data_folder = data_folder
        self.dest_folder = dest_folder

    def __call__(self, *args, **kwargs):
        for img_id in tqdm.tqdm(self.ids):
            img_path = os.path.join(self.data_folder, f'{img_id}.jpg')
            img = read_image(img_path, format="BGR")  # H, W, C
            img_height = img.shape[0]
            img_width = img.shape[1]

            # Get bboxes and classes GT
            bboxes_path = os.path.join(self.data_folder, f'{img_id}.txt')
            bboxes = np.loadtxt(bboxes_path, delimiter=" ", dtype=np.float32)
            if bboxes.ndim == 2:
                classes = bboxes[:, 0].tolist()
                bboxes = bboxes[:, 1:]
            else:  # only 1 instance
                classes = [bboxes[0].tolist()]
                bboxes = [bboxes[1:]]

            # Convert bboxes from YOLO format to Pascal VOC format
            bboxes = yolo_bboxes_to_pascal_voc(bboxes, img_height=img_height, img_width=img_width)

            predictions = self.predictor(img, bboxes, classes)

            instances = predictions["instances"].to(torch.device("cpu"))
            masks = instances.pred_masks.numpy()
            masks = np.array(masks).transpose((1, 2, 0))  # n x H x W -> H x W x n
            save_masks(dest_folder=self.dest_folder, filename=f'{img_id}.npz', masks=masks)

    def reload_weights_predictor(self):
        self.predictor.load_weights()
