import logging

import torch
import wandb
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model

from utils.albumentations_mapper import AlbumentationsMapper
from utils.setup_new_dataset import setup_new_dataset
from train_net import setup

logger = logging.getLogger("detectron2")


def run_on_image(inputs, outputs, best_res, worst_res):
    image = inputs["image"].permute(1, 2, 0).to("cpu").numpy()
    image = image[:, :, ::-1]  # BGR to RGB
    best_res.append(image)
    return best_res, worst_res


def log_selected_images(best_res, worst_res):
    best_img = []
    worst_img = []
    for img in best_res:
        best_img.append(wandb.Image(img, caption="example caption"))
    wandb.log({"examples": best_img})


def compute_best_and_worst_examples(args):
    # Init Weight & Biases and sync with Tensorboard
    wandb.init(project="Mask_RCNN", sync_tensorboard=True)

    cfg = setup(args)

    # Load model
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()

    # Setup dataset, mapper and dataloader
    setup_new_dataset()
    dataset_name = cfg.DATASETS.TEST[0]
    mapper = AlbumentationsMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)


    best_res = []
    worst_res = []
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
            best_res, worst_res = run_on_image(inputs[0], outputs[0], best_res, worst_res)

    log_selected_images(best_res, worst_res)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    compute_best_and_worst_examples(args)
