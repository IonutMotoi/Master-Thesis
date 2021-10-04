import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator


class PascalVOCEvaluator(DatasetEvaluator):
    def __init__(self):
        self._cpu_device = torch.device("cpu")
        self._predictions = []
        self._results = OrderedDict()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {
                "image_id": input["image_id"],
                "instances": []
            }
            instances = output["instances"].to(self._cpu_device)

            num_instance = len(instances)
            if num_instance != 0:
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist()

                for k in range(num_instance):
                    instance = {
                        "image_id": input["image_id"],
                        "category_id": classes[k],
                        "bbox": boxes[k],
                        "score": scores[k]
                    }
                    if instances.has("pred_masks"):
                        instance["segmentation"] = instances.pred_masks[k]
                    prediction["instances"].append(instance)

            self._predictions.append(prediction)

    def evaluate(self):
        # print(len(self._predictions))

        predictions = self._predictions

        self._results = OrderedDict()

        all_instances = list(itertools.chain(*[prediction["instances"] for prediction in predictions]))
        print(len(all_instances))

        # Get tasks from predictions
        tasks = ["bbox"]
        if "segmentation" in predictions[0]["instances"]:
            tasks.append("segm")
        print(tasks)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
