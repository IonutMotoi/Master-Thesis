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

from pseudo_labeling.mask_processing import dilate_pseudomasks, slic_pseudomasks, process_pseudomasks
from pseudo_labeling.masks_from_bboxes import generate_masks_from_bboxes
from sweep.sweep_utils import set_config_from_sweep, get_hyperparameters
from utils.early_stopping import EarlyStopping
from utils.setup_new_dataset import setup_new_dataset
from utils.setup_wgisd import setup_wgisd
from utils.albumentations_mapper import AlbumentationsMapper
from utils.loss import ValidationLossEval, MeanTrainLoss
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
    return results


def do_train(cfg, model, resume=False, model_weights=None):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    if model_weights is not None:
        start_iter = (checkpointer.resume_or_load(model_weights, resume=resume).get("iteration", -1) + 1)
    else:
        start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    max_iter = cfg.SOLVER.MAX_ITER
    # periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter, max_to_keep=1)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    mapper = AlbumentationsMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    examples_count = 0  # Counter for saving examples of augmented images on W&B
    validation_loss_eval_wgisd = ValidationLossEval(cfg, model, "wgisd_valid")
    validation_loss_eval_new_dataset = ValidationLossEval(cfg, model, "new_dataset_validation")
    mean_train_loss = MeanTrainLoss()
    early_stopping = EarlyStopping(patience=10)
    iters_per_epoch = cfg.SOLVER.ITERS_PER_EPOCH
    epoch = 0

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, start_iter+max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            mean_train_loss.update(loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            # periodic_checkpointer.step(iteration)

            # At the end of each epoch
            if (iteration - start_iter + 1) % iters_per_epoch == 0:
                # Train loss averaged over the epoch
                with storage.name_scope("Train losses"):
                    storage.put_scalars(total_loss=mean_train_loss.get_total_loss(),
                                        **mean_train_loss.get_losses(),
                                        smoothing_hint=False)
                mean_train_loss.reset()

                # Visualize some examples of augmented images and annotations
                if examples_count < 10:
                    image = visualize_image_and_annotations(data[0])
                    storage.put_image("Example of augmented image", image)
                    examples_count += 1

                # COCO Evaluation
                test_results = do_test(cfg, model)
                for dataset_name, dataset_results in test_results.items():
                    for name, results in dataset_results.items():
                        with storage.name_scope(f"{dataset_name}_{name}"):
                            storage.put_scalars(**results, smoothing_hint=False)

                # Validation loss
                validation_loss_dict_wgisd = validation_loss_eval_wgisd.get_loss()
                validation_loss_wgisd = sum(loss for loss in validation_loss_dict_wgisd.values())
                with storage.name_scope("Validation losses wgisd"):
                    storage.put_scalars(total_validation_loss_wgisd=validation_loss_wgisd,
                                        **validation_loss_dict_wgisd,
                                        smoothing_hint=False)
                logger.info("Total validation loss -> wgisd: {}".format(validation_loss_wgisd))

                validation_loss_dict_new_dataset = validation_loss_eval_new_dataset.get_loss()
                validation_loss_new_dataset = sum(loss for loss in validation_loss_dict_new_dataset.values())
                with storage.name_scope("Validation losses new dataset"):
                    storage.put_scalars(total_validation_loss_new_dataset=validation_loss_new_dataset,
                                        **validation_loss_dict_new_dataset,
                                        smoothing_hint=False)
                logger.info("Total validation loss -> new dataset: {}".format(validation_loss_new_dataset))

                comm.synchronize()

                # Write events to EventStorage
                for writer in writers:
                    writer.write()

                # Early stopping
                print("#######################################")
                metric = test_results["new_dataset_validation"]["segm"]["AP"]
                early_stopping.on_epoch_end(metric, epoch)
                storage.put_scalar(name='best_segm_AP', value=early_stopping.best_value, smoothing_hint=False)
                if early_stopping.has_improved:
                    print(f"New best model -> epoch: {epoch} -> segm AP: {metric}")
                    checkpointer.save("best_model")
                print("#######################################")

                if early_stopping.should_stop():
                    print(f"Early stopping at epoch {epoch}")
                    print(f"Best model was at epoch {early_stopping.best_epoch} "
                          f"with {early_stopping.best_value} segm AP")
                    break

                epoch += 1


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # to allow merging new keys
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Get default hyperparameters (can be over-ridden by a sweep)
    hyperparameters = get_hyperparameters(cfg)

    # Init Weight & Biases and sync with Tensorboard
    wandb.init(project="Mask_RCNN", sync_tensorboard=True, config=hyperparameters)

    cfg = set_config_from_sweep(cfg, wandb.config)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # Save config.yaml on wandb
    wandb.save(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), policy='now')

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    # Pseudo-masks will be generated in a subfolder of OUTPUT_DIR
    pseudo_masks_folders = []
    for folder in cfg.PSEUDOMASKS.DEST_FOLDER:
        pseudo_masks_folders.append(os.path.join(cfg.OUTPUT_DIR, folder))

    # Register dataset
    setup_new_dataset(pseudo_masks_folders[0])
    setup_wgisd()  # pass pseudo_masks_folders[1] to register pseudo_masks for wgisd (check cfg as well)

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

    for train_round in range(1, cfg.SOLVER.MAX_TRAINING_ROUNDS+1):
        print("#######################################")
        print(f"Training round {train_round} out of {cfg.SOLVER.MAX_TRAINING_ROUNDS}")
        print("#######################################")

        if train_round == 1:
            model_weights = None
        else:
            model_weights = os.path.join(cfg.OUTPUT_DIR, f"best_model_train_round_{train_round - 1}.pth")

        # Generate pseudo-masks
        if cfg.PSEUDOMASKS.GENERATE:
            for i in range(len(cfg.PSEUDOMASKS.IDS_TXT)):
                print(f"Generating pseudo-masks for dataset {i+1} out of {len(cfg.PSEUDOMASKS.IDS_TXT)}...")
                generate_masks_from_bboxes(cfg,
                                           ids_txt=cfg.PSEUDOMASKS.IDS_TXT[i],
                                           data_folder=cfg.PSEUDOMASKS.DATA_FOLDER[i],
                                           dest_folder=pseudo_masks_folders[i],
                                           model_weights=model_weights)

        # Post-process pseudo-masks
        if cfg.PSEUDOMASKS.PROCESS_METHOD in ['dilation', 'slic']:
            for i in range(len(cfg.PSEUDOMASKS.IDS_TXT)):
                print(f"Applying post-processing with {cfg.PSEUDOMASKS.PROCESS_METHOD} method to the pseudo-masks "
                      f"of dataset {i+1} out of {len(cfg.PSEUDOMASKS.IDS_TXT)}...")
                process_pseudomasks(cfg,
                                    method=cfg.PSEUDOMASKS.PROCESS_METHOD,
                                    input_masks=[f'{pseudo_masks_folders[i]}/*.npz'],
                                    data_path=cfg.PSEUDOMASKS.DATA_FOLDER[i],
                                    output_path=pseudo_masks_folders[i])

        # Train
        do_train(cfg, model, resume=args.resume, model_weights=model_weights)

        # Save the best model for each training round on Weight and Biases
        os.rename(os.path.join(cfg.OUTPUT_DIR, "best_model.pth"),
                  os.path.join(cfg.OUTPUT_DIR, f"best_model_train_round_{train_round}.pth"))
        wandb.save(os.path.join(cfg.OUTPUT_DIR, f"best_model_train_round_{train_round}.pth"), policy='now')
        # Need to remove the saved models due to limited space
        if train_round > 1:  # Remove previous model
            os.remove(os.path.join(cfg.OUTPUT_DIR, f"best_model_train_round_{train_round-1}.pth"))
        if train_round == cfg.SOLVER.MAX_TRAINING_ROUNDS:  # Remove last model
            os.remove(os.path.join(cfg.OUTPUT_DIR, f"best_model_train_round_{train_round}.pth"))

    return


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("-q", "--dry_run", action="store_true", help="Dry run (do not log to wandb)")
    args = parser.parse_args()

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
