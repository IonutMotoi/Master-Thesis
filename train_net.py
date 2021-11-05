import logging
import os
from collections import OrderedDict
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import launch, default_argument_parser, default_setup, PeriodicCheckpointer, default_writers
from detectron2.evaluation import inference_on_dataset, print_csv_format, COCOEvaluator
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.utils import comm
from detectron2.utils.events import EventStorage


from utils.setup_wgisd import setup_wgisd
from utils.albumentations_mapper import AlbumentationsMapper
from utils.validation_loss_eval import ValidationLossEval
from utils.visualization import visualize_image_and_annotations
from utils.pascal_voc_evaluator import PascalVOCEvaluator

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    if cfg.EVALUATOR == "pascal":
        if "detection" in dataset_name:
            task = "detection"
        else:
            task = "segmentation"
        return PascalVOCEvaluator(dataset_name, task)
    else:
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        mapper = AlbumentationsMapper(cfg, is_train=False)
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            if cfg.EVALUATOR == "pascal":
                evaluator.print_results(results_i)
            else:
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter, max_to_keep=1)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    mapper = AlbumentationsMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    examples_count = 0  # Counter for saving examples of augmented images on W&B
    validation_loss_eval = ValidationLossEval(cfg, model)

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                with storage.name_scope("Train losses"):
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if(
                cfg.TEST.EVAL_PERIOD > 0
                and ((iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 or iteration == max_iter - 1)
            ):
                test_results = do_test(cfg, model)
                for name, results in test_results.items():
                    with storage.name_scope(name):
                        storage.put_scalars(**results)

                validation_loss_dict = validation_loss_eval.get_loss()
                validation_loss = sum(loss for loss in validation_loss_dict.values())
                with storage.name_scope("Validation losses"):
                    storage.put_scalars(total_validation_loss=validation_loss, **validation_loss_dict)
                logger.info("Total validation loss: {}".format(validation_loss))

                comm.synchronize()

            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()

            # Visualize some examples of augmented images and annotations
            if examples_count < 5:
                image = visualize_image_and_annotations(data[0])
                storage.put_image("Example of augmented image", image)
                examples_count += 1

            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # to allow merging new keys
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # Init Weight & Biases and sync with Tensorboard
    wandb.init(project="Mask_RCNN", sync_tensorboard=True)

    cfg = setup(args)
    # Save config
    wandb.save(os.path.join(cfg.OUTPUT_DIR, "config.yaml"))

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    # Register dataset
    setup_wgisd()

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # Evaluate
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    # Train
    do_train(cfg, model, resume=args.resume)

    # Save final model
    wandb.save(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))

    return


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
