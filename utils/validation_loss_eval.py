import torch
from detectron2.data import build_detection_test_loader

from utils.albumentations_mapper import AlbumentationsMapper


def _dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = float(sum(d[key] for d in dict_list)) / len(dict_list)
    return mean_dict


class ValidationLossEval:
    def __init__(self, cfg, model):
        self.model = model
        dataset_name = cfg.DATASETS.TEST
        mapper = AlbumentationsMapper(cfg, is_train=False)
        self.data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def get_loss(self):
        losses = []
        for idx, inputs in enumerate(self.data_loader):
            with torch.no_grad():
                loss_dict = self.model(inputs)
                loss_dict = {k: v.item() for k, v in loss_dict.items()}
            losses.append(loss_dict)
        mean_loss_dict = _dict_mean(losses)
        return mean_loss_dict
