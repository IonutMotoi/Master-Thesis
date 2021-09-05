class AlbumentationsMapper:
    def __init__(self, cfg, is_train=True):
        self.aug = self._get_aug()
        pass

    def _get_aug(self):
        pass

    def __call__(self, dataset_dict):
        pass
