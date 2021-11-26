import os.path

import numpy as np
import torch
import tqdm
from detectron2.data.detection_utils import read_image

from pseudo_labeling.predictor import MasksFromBboxesPredictor
from utils.bbox_conversion import yolo_bboxes_to_pascal_voc
from utils.save import save_masks


def generate_masks_from_bboxes(cfg, ids_txt, data_folder, dest_folder, load_from_checkpoint=False):
    with open(ids_txt, 'r') as f:
        # Read all lines in file
        lines = f.readlines()
        # Recover the items ids, removing the \n at the end
        ids = [line.rstrip() for line in lines]

    predictor = MasksFromBboxesPredictor(cfg, load_from_checkpoint=load_from_checkpoint)
    data_folder = data_folder
    dest_folder = dest_folder

    for img_id in tqdm.tqdm(ids):
        img_path = os.path.join(data_folder, f'{img_id}.jpg')
        img = read_image(img_path, format="BGR")  # H, W, C
        img_height = img.shape[0]
        img_width = img.shape[1]

        # Get bboxes and classes GT
        bboxes_path = os.path.join(data_folder, f'{img_id}.txt')
        bboxes = np.loadtxt(bboxes_path, delimiter=" ", dtype=np.float32)
        if bboxes.ndim == 2:
            classes = bboxes[:, 0].tolist()
            bboxes = bboxes[:, 1:]
        else:  # only 1 instance
            classes = [bboxes[0]]
            bboxes = [bboxes[1:]]

        # Convert bboxes from YOLO format to Pascal VOC format
        bboxes = yolo_bboxes_to_pascal_voc(bboxes, img_height=img_height, img_width=img_width)

        predictions = predictor(img, bboxes, classes)

        instances = predictions["instances"].to(torch.device("cpu"))
        masks = instances.pred_masks.numpy()
        masks = np.array(masks).transpose((1, 2, 0))  # n x H x W -> H x W x n
        save_masks(dest_folder=dest_folder, filename=f'{img_id}.npz', masks=masks)
