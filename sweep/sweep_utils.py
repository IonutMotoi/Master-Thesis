def get_hyperparameters(cfg):
    """
    Get the hyperparameters that we want to pass to wandb in order to do a hyperparameter sweep
    :param cfg: detectron2 cfg
    :return: (dict) Hyperparameters
    """
    hyperparameters = dict(
        iterative_pseudomasks_period=cfg.ITERATIVE_PSEUDOMASKS.PERIOD
    )
    return hyperparameters


def set_config_from_sweep(cfg, hyperparameters):
    if cfg.is_frozen():
        cfg.defrost()
    cfg.ITERATIVE_PSEUDOMASKS.PERIOD = hyperparameters.iterative_pseudomasks_period
    return cfg