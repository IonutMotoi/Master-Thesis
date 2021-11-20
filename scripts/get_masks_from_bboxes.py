from pathlib import Path

from detectron2.utils.logger import setup_logger

from utils.inference_setup import get_parser, setup
from pseudo_labeling.masks_from_bboxes import MasksFromBboxes


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--ids",
                        default="/thesis/new_dataset/train/train.txt",
                        help="Path of the txt file containing the ids of the images")
    parser.add_argument("--data",
                        default="/thesis/new_dataset/train",
                        help="Path of the folder containing the data")
    parser.add_argument("--dest",
                        default="./pseudo_labels",
                        help="Path where to save the pseudo masks")
    args = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    # Create destination folder
    Path(args.dest).mkdir(parents=True, exist_ok=True)

    masks_from_bboxes = MasksFromBboxes(cfg, ids_txt=args.ids, data_folder=args.data, dest_folder=args.dest)
    masks_from_bboxes.get_masks_from_bboxes()
