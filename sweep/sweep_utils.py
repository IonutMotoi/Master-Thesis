def get_hyperparameters(cfg):
    """
    Get the hyperparameters that we want to pass to wandb in order to do a hyperparameter sweep
    :param cfg: detectron2 cfg
    :return: (dict) Hyperparameters
    """
    hyperparameters = dict(
        mask_process_method=cfg.PSEUDOMASKS.PROCESS_METHOD,
        max_training_rounds=cfg.SOLVER.MAX_TRAINING_ROUNDS,
        model_weights=cfg.MODEL.WEIGHTS
    )
    return hyperparameters


def set_config_from_sweep(cfg, sweep_params):
    # Overwrite config with parameters from a wandb sweep, if any, otherwise set the default values from config
    assert not cfg.is_frozen()
    cfg.PSEUDOMASKS.PROCESS_METHOD = sweep_params.mask_process_method
    cfg.SOLVER.MAX_TRAINING_ROUNDS = sweep_params.max_training_rounds
    cfg.MODEL.WEIGHTS = sweep_params.model_weights
    if 'model_final_a3ec72.pkl' not in sweep_params.model_weights:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_finetuning'
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_' + sweep_params.mask_process_method
    return cfg
