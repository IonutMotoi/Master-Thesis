from collections import OrderedDict, defaultdict
import numpy as np
import torch
from pycocotools.mask import decode

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator


class PascalVOCEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC 2007 style AP for detection and instance segmentation on a custom dataset.
    """

    def __init__(self, dataset_name, task):
        meta = MetadataCatalog.get(dataset_name)
        self.num_of_classes = len(meta.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.results = OrderedDict()
        self.predictions = None  # initialized inside reset()
        self.annotations = None  # initialized inside reset()
        self.task = task

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
                if instances.has("pred_masks"):
                    prediction["mask"] = instances.pred_masks[k].numpy()
                self.predictions[classes[k]].append(prediction)

    def evaluate(self):
        if self.task == "detection":
            ious = list(range(30, 85, 10))
        else:
            ious = list(range(30, 95, 10))

        aps = defaultdict(list)  # iou -> ap per class
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        f1s = defaultdict(list)
        for class_id in range(self.num_of_classes):
            for threshold in ious:
                ap, precision, recall, f1 = self.voc_eval(class_id, threshold / 100.0)
                aps[threshold].append(ap)
                precisions[threshold].append(precision)
                recalls[threshold].append(recall)
                f1s[threshold].append(f1)

        # Average over all classes
        aps = {iou: np.round(np.mean(x), 3) for iou, x in aps.items()}
        precisions = {iou: np.round(np.mean(x), 3) for iou, x in precisions.items()}
        recalls = {iou: np.round(np.mean(x), 3) for iou, x in recalls.items()}
        f1s = {iou: np.round(np.mean(x), 3) for iou, x in f1s.items()}

        ret = OrderedDict()
        ret["bbox"] = {
            "IoU": ious,
            "AP": aps,
            "Precision": precisions,
            "Recall": recalls,
            "F1": f1s
        }

        return ret

    def voc_eval(self, class_id, overlap_threshold):
        if self.task == "detection":
            return self.voc_eval_detection(class_id, overlap_threshold)
        else:
            return self.voc_eval_instance_segmentation(class_id, overlap_threshold)

    def voc_eval_detection(self, class_id, overlap_threshold):
        """ recall, precision, ap = voc_eval(class_id, overlap_threshold) """
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
            bbox = bboxes[i, :]
            overlap_max = -np.inf

            if bboxes_gt.size > 0:
                # compute overlaps

                # intersection
                ixmin = np.maximum(bboxes_gt[:, 0], bbox[0])
                iymin = np.maximum(bboxes_gt[:, 1], bbox[1])
                ixmax = np.minimum(bboxes_gt[:, 2], bbox[2])
                iymax = np.minimum(bboxes_gt[:, 3], bbox[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih

                # union
                uni = ((bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0)
                       + (bboxes_gt[:, 2] - bboxes_gt[:, 0] + 1.0) * (bboxes_gt[:, 3] - bboxes_gt[:, 1] + 1.0)
                       - inters)

                overlaps = inters / uni
                overlap_max = np.max(overlaps)
                j_max = np.argmax(overlaps)

            if overlap_max > overlap_threshold:
                if not img_annotations["det"][j_max]:
                    tp[i] = 1.0
                    img_annotations["det"][j_max] = 1
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute precision and recall
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / float(npos)  # npos == tp + fn
        precision = tp / (tp + fp)

        # Compute AP
        ap = self.voc_ap(recall, precision)

        recall = recall[-1]
        precision = precision[-1]

        # Compute F1
        f1 = 2 * precision * recall / (precision + recall)

        return ap, precision, recall, f1

    def voc_eval_instance_segmentation(self, class_id, overlap_threshold):
        npos = 0
        # Get annotations of class_id
        annotations = {}  # image id -> (dict) annotations of class_id
        for image_id, image_annotations in self.annotations.items():
            image_class_annotations = [annotation for annotation in image_annotations
                                       if annotation["category_id"] == class_id]
            masks = np.array([decode(annotation["segmentation"]) for annotation in image_class_annotations])
            det = [False] * len(image_class_annotations)
            npos += len(image_class_annotations)
            annotations[image_id] = {"masks": masks, "det": det}

        # Get predictions of class_id
        predictions = self.predictions[class_id]
        image_ids = [prediction["image_id"] for prediction in predictions]
        confidence = np.array([prediction["score"] for prediction in predictions])
        masks = np.array([prediction["mask"] for prediction in predictions])

        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidence)[::-1]
        masks = masks[sorted_indices]
        image_ids = [image_ids[x] for x in sorted_indices]

        # Get TPs and FPs
        num_of_predictions = len(image_ids)
        tp = np.zeros(num_of_predictions)
        fp = np.zeros(num_of_predictions)
        for i in range(num_of_predictions):
            img_annotations = annotations[image_ids[i]]
            masks_gt = img_annotations["masks"]
            mask = masks[i]
            overlap_max = -np.inf

            if masks_gt.size > 0:
                # intersection
                inters = np.sum((masks_gt * mask) > 0, axis=(1, 2))

                # union
                uni = np.sum((masks_gt + mask) > 0, axis=(1, 2))

                # compute overlaps
                overlaps = inters / uni
                overlap_max = np.max(overlaps)
                j_max = np.argmax(overlaps)

            if overlap_max > overlap_threshold:
                if not img_annotations["det"][j_max]:
                    tp[i] = 1.0
                    img_annotations["det"][j_max] = 1
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute precision and recall
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / float(npos)  # npos == tp + fn
        precision = tp / (tp + fp)

        # Compute AP
        ap = self.voc_ap(recall, precision)

        recall = recall[-1]
        precision = precision[-1]

        # Compute F1
        f1 = 2 * precision * recall / (precision + recall)

        return ap, precision, recall, f1

    def voc_ap(self, recall, precision):
        """
        Compute VOC AP given precision and recall using the VOC 07 11-point method.
        """
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.0
        return ap

    def print_results(self, ret):
        if self.task == "detection":
            print("Results for detection:", '\n')
        else:
            print("Results for instance segmentation", '\n')

        print("IoU:")
        print(ret["bbox"]["IoU"], '\n')

        print("AP:")
        print([x for x in ret["bbox"]["AP"].values()], '\n')

        print("Precision:")
        print([x for x in ret["bbox"]["Precision"].values()], '\n')

        print("Recall:")
        print([x for x in ret["bbox"]["Recall"].values()], '\n')

        print("F1:")
        print([x for x in ret["bbox"]["F1"].values()], '\n')
