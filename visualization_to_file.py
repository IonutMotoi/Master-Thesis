import glob
import os
import time
from pathlib import Path

import tqdm
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode

from utils.visualization import Visualization
from utils.inference_setup import setup, get_parser


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--detection-only", action='store_true', help="Show only the bounding boxes")
    parser.add_argument("--random-colors", action='store_true', help="Random colors for each instance")
    args = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    if cfg.RANDOM_COLORS:
        instance_mode = ColorMode.IMAGE
    else:
        instance_mode = ColorMode.SEGMENTATION
    visualization = Visualization(
        cfg,
        detection_only=args.detection_only,
        instance_mode=instance_mode)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()

        predictions, visualized_output = visualization.run_on_image(img)

        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        Path(args.output).mkdir(parents=True, exist_ok=True)  # Create destination folder
        assert os.path.isdir(args.output), args.output
        out_filename = os.path.join(args.output, os.path.basename(path))
        visualized_output.save(out_filename)
