import albumentations as A
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode, Visualizer

from utils.albumentations_mapper import get_augmentations


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


class Predictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input.
    3. Apply resizing and padding defined by `cfg.ALBUMENTATIONS.LONGEST_MAX_SIZE' and 'cfg.ALBUMENTATIONS.PAD`.
    4. Take one input image and produce a single output together with the transformed image.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = Predictor(cfg)
        input = cv2.imread("input.jpg")
        outputs, image = pred(input)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform = A.Compose(get_augmentations(cfg, is_train=False))

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == "BGR", self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
            image: transformed image (H, W, C) (BGR format)
        """
        with torch.no_grad():
            # Albumentations expects to receive an image in the RGB format
            original_image = original_image[:, :, ::-1]
            # Apply transformations
            transformed = self.transform(image=original_image)
            image = transformed["image"]
            # The model expects BGR inputs
            image = image[:, :, ::-1]
            height, width = image.shape[:2]
            # Convert H,W,C image to C,H,W tensor
            tensor_image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": tensor_image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions, image


def visualize_image_and_annotations(data):
    image = data["image"]
    image = image[[2, 1, 0], :, :]  # BGR to RGB
    image = image.permute(1, 2, 0)  # torch.tensor C,W,H to W,H,C
    visualizer = Visualizer(image)
    out = visualizer.overlay_instances(boxes=data["instances"].gt_boxes, masks=data["instances"].gt_masks)
    image = out.get_image()
    image = image.transpose(2, 0, 1)  # ndarray W,H,C to C,W,H
    return image