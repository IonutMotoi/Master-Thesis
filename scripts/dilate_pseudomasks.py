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


def dilate_pseudomasks(input_masks, path_bboxes, output_path):
    kernel = get_default_kernel()

    if len(input_masks) == 1:
        input_masks = glob.glob(os.path.expanduser(input_masks))
        assert input_masks, "The input path(s) was not found"
    for path in tqdm.tqdm(input_masks, disable=not output_path):
        masks = np.load(path)['arr_0'].astype(np.uint8)  # H x W x n

        masks_height = masks.shape[0]
        masks_width = masks.shape[1]

        masks_id = os.path.basename(path)
        masks_id = os.path.splitext(masks_id)[0]
        print(masks_id)

        bboxes = np.loadtxt(os.path.join(path_bboxes, f'{masks_id}.txt'), delimiter=" ", dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
        bboxes = bboxes[:, 1:]  # remove classes

        # Dilate the masks until they touch the edges of the bounding boxes
        for i in range(masks.shape[2]):
            if np.all((masks[:, :, i] == 0)):  # if empty mask
                continue
            abs_bbox = yolo_bbox_to_pascal_voc(bboxes[i], img_height=masks_height, img_width=masks_width)
            while not mask_touches_bbox(masks[:, :, i], abs_bbox, touches_all_edges=False):
                masks[:, :, i] = cv2.dilate(masks[:, :, i], kernel, iterations=1)
            set_values_outside_bbox_to_zero(masks[:, :, i], abs_bbox)

        # Create destination folder
        Path(output_path).mkdir(parents=True, exist_ok=True)
        save_masks(masks=masks, dest_folder=output_path, filename=f'{masks_id}.npz')


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--path-bboxes",
        default="/thesis/new_dataset/train",
        help="path to the bboxes txt files",
    )
    args = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    dilate_pseudomasks(input_masks=args.input, path_bboxes=args.path_bboxes, output_path=args.output)
