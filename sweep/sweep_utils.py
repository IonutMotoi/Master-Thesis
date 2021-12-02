def get_hyperparameters(cfg):
    """
    Get the hyperparameters that we want to pass to wandb in order to do a hyperparameter sweep
    :param cfg: detectron2 cfg
    :return: (dict) Hyperparameters
    """
    hyperparameters = dict(
        pseudomasks_period=cfg.PSEUDOMASKS.PERIOD
    )
    return hyperparameters


def set_config_from_sweep(cfg, sweep):
    assert not cfg.is_frozen()
    cfg.PSEUDOMASKS.PERIOD = sweep.pseudomasks_period
    return cfg
