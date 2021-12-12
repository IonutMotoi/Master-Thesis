def get_hyperparameters(cfg):
    """
    Get the hyperparameters that we want to pass to wandb in order to do a hyperparameter sweep
    :param cfg: detectron2 cfg
    :return: (dict) Hyperparameters
    """
    hyperparameters = dict(
        mask_process_method=cfg.PSEUDOMASKS.PROCESS_METHOD,
        max_training_rounds=cfg.SOLVER.MAX_TRAINING_ROUNDS,
        model_weights=cfg.MODEL.WEIGHTS,
        slic_zero=cfg.PSEUDOMASKS.SLIC.SLIC_ZERO,
        n_segments=cfg.PSEUDOMASKS.SLIC.N_SEGMENTS,
        compactness=cfg.PSEUDOMASKS.SLIC.COMPACTNESS,
        sigma=cfg.PSEUDOMASKS.SLIC.SIGMA,
        threshold=cfg.PSEUDOMASKS.SLIC.THRESHOLD
    )
    return hyperparameters


def set_config_from_sweep(cfg, sweep_params):
    # Overwrite config with parameters from a wandb sweep, if any, otherwise set the default values from config
    assert not cfg.is_frozen()
    cfg.PSEUDOMASKS.PROCESS_METHOD = sweep_params.mask_process_method
    cfg.SOLVER.MAX_TRAINING_ROUNDS = sweep_params.max_training_rounds
    cfg.MODEL.WEIGHTS = sweep_params.model_weights
    cfg.PSEUDOMASKS.SLIC.N_SEGMENTS = sweep_params.n_segments
    cfg.PSEUDOMASKS.SLIC.SLIC_ZERO = sweep_params.slic_zero
    cfg.PSEUDOMASKS.SLIC.COMPACTNESS = sweep_params.compactness
    cfg.PSEUDOMASKS.SLIC.SIGMA = sweep_params.sigma
    cfg.PSEUDOMASKS.SLIC.THRESHOLD = sweep_params.threshold
    if 'model_final_a3ec72.pkl' not in sweep_params.model_weights:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_finetuning'
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_' + cfg.PSEUDOMASKS.PROCESS_METHOD
    if cfg.PSEUDOMASKS.PROCESS_METHOD == 'slic':
        cfg.OUTPUT_DIR = (cfg.OUTPUT_DIR
                          + '_segments' + str(cfg.PSEUDOMASKS.SLIC.N_SEGMENTS)
                          + '_compactness' + str(cfg.PSEUDOMASKS.SLIC.COMPACTNESS)
                          + '_sliczero' + str(cfg.PSEUDOMASKS.SLIC.SLIC_ZERO)
                          + '_threshold' + str(cfg.PSEUDOMASKS.SLIC.THRESHOLD))
    return cfg
