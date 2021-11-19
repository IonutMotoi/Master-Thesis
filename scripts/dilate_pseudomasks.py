import glob
import os
from pathlib import Path

import cv2
import numpy as np
from detectron2.utils.logger import setup_logger
import tqdm

from image_processing.mask_processing import get_default_kernel, mask_touches_bbox, set_values_outside_bbox_to_zero
from utils.bbox_conversion import yolo_bbox_to_pascal_voc
from utils.inference_setup import get_parser, setup
from utils.save import save_masks

if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    kernel = get_default_kernel()

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        masks = np.load(path)['arr_0'].astype(np.uint8)  # H x W x n

        masks_height = masks.shape[0]
        masks_width = masks.shape[1]

        masks_id = os.path.basename(path)
        masks_id = os.path.splitext(masks_id)[0]

        path_bboxes = "/thesis/new_dataset/train"
        bboxes = np.loadtxt(os.path.join(path_bboxes, f'{masks_id}.txt'), delimiter=" ", dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)

        # Dilate the masks until they touch the edges of the bounding boxes
        for i in range(masks.shape[2]):
            abs_bbox = yolo_bbox_to_pascal_voc(bboxes[i], img_height=masks_height, img_width=masks_width)
            while not mask_touches_bbox(masks[:, :, i], abs_bbox, touches_all_edges=False):
                masks[:, :, i] = cv2.dilate(masks[:, :, i], kernel, iterations=1)
            set_values_outside_bbox_to_zero(masks[:, :, i], abs_bbox)

        # Create destination folder
        Path(args.output).mkdir(parents=True, exist_ok=True)
        save_masks(masks=masks, dest_folder=args.output, filename=f'{masks_id}.npz')
