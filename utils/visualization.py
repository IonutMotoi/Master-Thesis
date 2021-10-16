import torch

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from utils.predictor import Predictor


class Visualization(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = Predictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions, image = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


def visualize_image_and_annotations(data):
    image = data["image"]
    image = image[[2, 1, 0], :, :]  # BGR to RGB
    image = image.permute(1, 2, 0)  # torch.tensor C,W,H to W,H,C
    visualizer = Visualizer(image)
    out = visualizer.overlay_instances(boxes=data["instances"].gt_boxes, masks=data["instances"].gt_masks)
    image = out.get_image()
    image = image.transpose(2, 0, 1)  # ndarray W,H,C to C,W,H
    return image