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
import sys

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
        self.num_of_classes = len(meta.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.results = OrderedDict()
        self.predictions = None  # initialized inside reset()
        self.annotations = None

    def reset(self):
        self.predictions = defaultdict(list)  # class id -> (list of dicts) predictions
        self.annotations = {}  # image id -> (list of dicts) annotations - ground truth

    def process(self, inputs, outputs):
        for input_, output in zip(inputs, outputs):
            image_id = input_["image_id"]

            # Get annotations ground truth
            self.annotations[image_id] = input_["annotations"]

            # Get predictions
            instances = output["instances"].to(self.cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.tolist()  # category id

            for k in range(len(instances)):
                prediction = {
                    "image_id": image_id,
                    "category_id": classes[k],
                    "bbox": boxes[k],
                    "score": scores[k]
                }
                #    if instances.has("pred_masks"):
                #        prediction["segmentation"] = instances.pred_masks[k]
                self.predictions[classes[k]].append(prediction)

    def evaluate(self):
        aps = defaultdict(list)  # iou -> ap per class
        for class_id in range(self.num_of_classes):
            for threshold in range(50, 100, 5):
                recall, precision, ap = self.voc_eval(class_id, threshold)
                aps[threshold].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        return ret

    def voc_eval(self, class_id, overlap_threshold=0.5):
        """rec, prec, ap = voc_eval(class_id, [ovthresh])"""
        npos = 0
        # Get annotations of class_id
        annotations = {}  # image id -> (dict) annotations of class_id
        for image_id, image_annotations in self.annotations.items():
            image_class_annotations = [annotation for annotation in image_annotations
                                       if annotation["category_id"] == class_id]
            bboxes = np.array([annotation["bbox"] for annotation in image_class_annotations])
            det = [False] * len(image_class_annotations)
            npos += len(image_class_annotations)
            annotations[image_id] = {"bboxes": bboxes, "det": det}


        # Get predictions of class_id
        predictions = self.predictions[class_id]
        image_ids = [prediction["image_id"] for prediction in predictions]
        confidence = np.array([prediction["score"] for prediction in predictions])
        bboxes = np.array([prediction["bbox"] for prediction in predictions])

        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidence)[::-1]
        bboxes = bboxes[sorted_indices, :]
        image_ids = [image_ids[x] for x in sorted_indices]

        # Get TPs and FPs
        num_of_predictions = len(image_ids)
        tp = np.zeros(num_of_predictions)
        fp = np.zeros(num_of_predictions)
        for i in range(num_of_predictions):
            img_annotations = annotations[image_ids[i]]
            bboxes_gt = img_annotations["bboxes"]
            print(bboxes_gt)
            bbox = bboxes[i, :]
            print(bbox)
            overlap_max = -np.inf
            sys.exit()

        # TODO: Compute precision and recall

        # TODO: Compute AP

        return 0, 0, 0
