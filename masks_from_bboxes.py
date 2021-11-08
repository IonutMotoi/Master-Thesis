import glob
import os
import time

import cv2
import numpy as np
import torch
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from utils.inference_setup import get_parser, setup
from utils.predictor import MasksFromBboxesPredictor
from utils.save import save_masks
from utils.bbox_conversion import pascal_voc_bboxes_to_yolo, yolo_bboxes_to_pascal_voc


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)
    predictor = MasksFromBboxesPredictor(cfg)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        image = read_image(path, format="BGR")  # H, W, C
        img_height = image.shape[0]
        img_width = image.shape[1]

        root = os.path.dirname(path)
        img_id = os.path.basename(path)
        img_id = os.path.splitext(img_id)[0]

        # Get bboxes and classes GT
        bboxes_path = os.path.join(root, f'{img_id}.txt')
        bboxes = np.loadtxt(bboxes_path, delimiter=" ", dtype=np.float32)
        classes = bboxes[:, 0].tolist()
        bboxes = bboxes[:, 1:]

        # Convert bboxes from YOLO format to Pascal VOC format
        bboxes = yolo_bboxes_to_pascal_voc(bboxes, img_height=img_height, img_width=img_width)

        start_time = time.time()
        predictions = predictor(image, bboxes, classes)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if len(predictions["instances"]) > 0:
            instances = predictions["instances"].to(torch.device("cpu"))
            masks = instances.pred_masks.numpy()

            # Save an example image with labels overlay
            image = image[:, :, ::-1]  # BGR to RGB
            visualizer = Visualizer(image)
            out = visualizer.overlay_instances(boxes=bboxes, masks=masks)
            image = out.get_image()
            image = image.transpose(2, 0, 1)  # ndarray W,H,C to C,W,H
            cv2.imwrite('sample_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # n x H x W -> H x W x n
            masks = np.array(masks).transpose((1, 2, 0))

            save_masks(dest_folder=args.output, filename=f'{img_id}.npz', masks=masks)
