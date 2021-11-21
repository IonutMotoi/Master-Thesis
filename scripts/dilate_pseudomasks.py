from detectron2.utils.logger import setup_logger

from image_processing.mask_processing import dilate_pseudomasks
from utils.inference_setup import get_parser

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
