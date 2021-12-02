def get_hyperparameters(cfg):
    """
    Get the hyperparameters that we want to pass to wandb in order to do a hyperparameter sweep
    :param cfg: detectron2 cfg
    :return: (dict) Hyperparameters
    """
    hyperparameters = dict(
        iterative_pseudomasks_period=cfg.ITERATIVE_PSEUDOMASKS.PERIOD,
    )
    return hyperparameters


def set_config_from_sweep(cfg, sweep):
    # Overwrite config with parameters from a wandb sweep, if any, otherwise set the default values from config
    assert not cfg.is_frozen()
    cfg.ITERATIVE_PSEUDOMASKS.PERIOD = sweep.iterative_pseudomasks_period
    return cfg
