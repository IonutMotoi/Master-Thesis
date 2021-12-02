def get_hyperparameters(cfg):
    """
    Get the hyperparameters that we want to pass to wandb in order to do a hyperparameter sweep
    :param cfg: detectron2 cfg
    :return: (dict) Hyperparameters
    """
    hyperparameters = dict(
        test=1
    )
    return hyperparameters


def set_config_from_hyperparameters(cfg, hyperparameters):
    if cfg.is_frozen():
        cfg.defrost()
    return cfg