import logging

import cv2
import torch
import wandb
import numpy as np
from detectron2.utils.events import EventStorage
from pycocotools.mask import decode

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model

from utils.albumentations_mapper import AlbumentationsMapper
from utils.setup_new_dataset import setup_new_dataset
from train_net import setup

logger = logging.getLogger("detectron2")


def run_on_image(inputs, mask_loss, outputs, sorted_results):
    gt_masks = [decode(anno["segmentation"]) for anno in inputs["annotations"]]
    gt_masks = np.stack(gt_masks)
    gt_masks = np.any(gt_masks, axis=0).astype(np.uint8)

    pred_masks = outputs["instances"].pred_masks.to("cpu").numpy()
    pred_masks = np.any(pred_masks, axis=0).astype(np.uint8)

    sample = {
        "image_id": inputs["image_id"],
        "file_name": inputs["file_name"],
        "gt_masks": gt_masks,
        "pred_masks": pred_masks,
        "mask_loss": mask_loss
    }

    sorted_results.append(sample)
    sorted_results.sort(key=lambda x: x["mask_loss"])

    return sorted_results


def log_selected_images(results, caption="", max_res=3):
    class_labels = {1: "grapes"}
    scale_factor = 0.25

    for i, sample in enumerate(results):
        image = cv2.imread(sample["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        pred_masks = sample["pred_masks"]
        pred_masks = cv2.resize(pred_masks, None, fx=scale_factor, fy=scale_factor)

        gt_masks = sample["gt_masks"]
        gt_masks = cv2.resize(gt_masks, None, fx=scale_factor, fy=scale_factor)

        wandb.Image(image,
                    masks={
                        "predictions": {
                            "mask_data": pred_masks,
                            "class_labels": class_labels
                        },
                        "ground_truth": {
                            "mask_data": gt_masks,
                            "class_labels": class_labels
                        }
                    },
                    caption=sample["image_id"])
        wandb.log({caption: [image]})

        if i == max_res-1:
            break


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

    # Setup dataset, mapper and dataloader
    setup_new_dataset()
    dataset_name = cfg.DATASETS.TEST[0]
    mapper = AlbumentationsMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    sorted_results = []
    with EventStorage():
        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                # Get mask loss
                model.train()
                loss_dict = model(inputs)
                mask_loss = loss_dict["loss_mask"].item()

                # Get predictions
                model.eval()
                outputs = model(inputs)

                sorted_results = run_on_image(inputs[0], mask_loss, outputs[0], sorted_results)

    log_selected_images(sorted_results, caption="Best examples", max_res=3)
    log_selected_images(sorted_results[::-1], caption="Worst examples", max_res=3)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    compute_best_and_worst_examples(args)
