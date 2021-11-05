import glob
import os
import time
from pathlib import Path
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from utils.inference_setup import get_parser, setup
from utils.predictor import MasksFromBboxesPredictor

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
        # use PIL, to be consistent with evaluation
        image = read_image(path, format="BGR")
        start_time = time.time()

        predictions = predictor(image)

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
