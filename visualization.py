from detectron2.utils.visualizer import Visualizer


def visualize_image_and_annotations(data):
    image = data["image"]
    image = image[[2, 1, 0], :, :]  # BGR to RGB
    image = image.permute(1, 2, 0)  # torch.tensor C,W,H to W,H,C
    visualizer = Visualizer(image)
    out = visualizer.overlay_instances(boxes=data["instances"].gt_boxes, masks=data["instances"].gt_masks)
    image = out.get_image()
    image = image.transpose(2, 0, 1)  # ndarray W,H,C to C,W,H
    return image