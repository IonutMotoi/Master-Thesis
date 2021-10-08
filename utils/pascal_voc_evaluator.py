import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict, defaultdict
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
    """
    Evaluate Pascal VOC 2007 style AP for detections and instance segmentation on a custom dataset.
    """

    def __init__(self, dataset_name):
        meta = MetadataCatalog.get(dataset_name)
        self.class_names = meta.thing_classes
        self.cpu_device = torch.device("cpu")
        self.results = OrderedDict()
        self.predictions = None  # initialized inside reset()
        self.annotations = None

    def reset(self):
        self.predictions = defaultdict(list)  # class name -> list of predictions
        self.annotations = {}  # image id -> dict with annotations (ground truth)

    def process(self, inputs, outputs):
        for input_, output in zip(inputs, outputs):
            image_id = input_["image_id"]

            # Get annotations ground truth
            self.annotations[image_id] = input_["annotations"]

            # Get predictions for each class
            instances = output["instances"].to(self.cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.tolist()
            print(classes)
            # for k in range(len(instances)):


            # prediction = {
            #     "image_id": input["image_id"],
            #     "instances": []
            # }
            # instances = output["instances"].to(self._cpu_device)
            #
            # num_instance = len(instances)
            # if num_instance != 0:
            #     boxes = instances.pred_boxes.tensor.numpy()
            #     scores = instances.scores.tolist()
            #     classes = instances.pred_classes.tolist()
            #
            #     for k in range(num_instance):
            #         instance = {
            #             "image_id": input["image_id"],
            #             "category_id": classes[k],
            #             "bbox": boxes[k],
            #             "score": scores[k]
            #         }
            #         if instances.has("pred_masks"):
            #             instance["segmentation"] = instances.pred_masks[k]
            #         prediction["instances"].append(instance)
            #
            # self._predictions.append(prediction)

    def evaluate(self):
        # predictions = self._predictions
        # self._results = OrderedDict()
        #
        # all_instances = list(itertools.chain(*[prediction["instances"] for prediction in predictions]))
        #
        # # Get tasks from predictions
        # tasks = ["bbox"]
        # if "segmentation" in all_instances[0]:
        #     tasks.append("segm")
        #
        # for task in tasks:
        #     self._results[task] = _evaluate_predictions(all_instances)
        #
        # # Copy so the caller can do whatever with results
        # return copy.deepcopy(self._results)
        pass


def mean_average_precision(pred_boxes, pred_classes, pred_scores, true_boxes, iou_threshold=0.5):
    average_precisions = []
