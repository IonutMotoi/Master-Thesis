import glob
import os

import numpy as np
from detectron2.utils.logger import setup_logger
import tqdm

from utils.inference_setup import get_parser, setup

if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

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
        if len(bboxes) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
        print(bboxes.shape)
