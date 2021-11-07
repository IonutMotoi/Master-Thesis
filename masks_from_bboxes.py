import glob
import os
import time
import numpy as np
import torch
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances, Boxes
from detectron2.utils.logger import setup_logger

from utils.inference_setup import get_parser, setup
from utils.predictor import MasksFromBboxesPredictor
from utils.save import save_image_and_labels
from utils.bbox_conversion import pascal_voc_bboxes_to_yolo

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

        # Create an 'Instances' object
        target = Instances(image_size=(img_height, img_width))
        target.pred_boxes = Boxes(bboxes)
        target.pred_classes = torch.tensor(classes, dtype=torch.int64)

        start_time = time.time()
        predictions = predictor(image, target)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        instances = predictions["instances"].to(torch.device("cpu"))
        bboxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy()

        # n x H x W -> H x W x n
        masks = np.array(masks).transpose((1, 2, 0))

        # Convert bboxes from Pascal VOC format to YOLO format
        bboxes = pascal_voc_bboxes_to_yolo(bboxes, img_height, img_width)
        # bboxes = np.array(bboxes)
        print("BBOXES", bboxes)
        print("CLASSES", classes)
        save_image_and_labels(
            dest_folder=args.output,
            img_id=img_id,
            image=image,
            class_labels=classes,
            bboxes=bboxes,
            masks=masks,
            img_format="BGR"
        )
