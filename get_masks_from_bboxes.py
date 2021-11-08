from detectron2.utils.logger import setup_logger

from utils.inference_setup import get_parser, setup
from pseudo_labeling.masks_from_bboxes import MasksFromBboxes



if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--ids",
                        required=True,
                        default="/thesis/wgisd/train_without_masked_train_and_valid_ids.txt",
                        help="Also save an image with labels overlay")
    parser.add_argument("--data",
                        required=True,
                        default="/thesis/wgisd/data",
                        help="Also save an image with labels overlay")
    parser.add_argument("--dest",
                        required=True,
                        default="./pseudo_labels",
                        help="Also save an image with labels overlay")
    args = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    masks_from_bboxes = MasksFromBboxes(cfg, ids_txt=args.ids, data_folder=args.data, dest_folder=args.dest)
    masks_from_bboxes()
