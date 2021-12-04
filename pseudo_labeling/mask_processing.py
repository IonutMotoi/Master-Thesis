import glob
import os
from pathlib import Path

import cv2
import numpy as np
import tqdm

from utils.bbox_conversion import yolo_bbox_to_pascal_voc
from utils.save import save_masks


def get_default_kernel():
    kernel = [[0, 1, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0]]
    return np.array(kernel).astype(np.uint8)


def mask_touches_bbox(mask, bbox, touches_all_edges=False):
    """
    Check if the mask touches the bounding box
    :param mask: Mask as np.array uint8
    :param bbox: Bounding box as np.array in Pascal VOC format (x_min, y_min, x_max, y_max)
    :param touches_all_edges: (default False)
    :return: If touches_all_edges=True then returns True if the mask touches all the edges of the bbox,
             else returns True if the mask touches at least one of the edges of the bbox
    """
    x = np.where(np.any(mask, axis=0))[0]
    y = np.where(np.any(mask, axis=1))[0]
    temp_bbox = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)
    if touches_all_edges:
        return (temp_bbox[0] <= bbox[0] and
                temp_bbox[1] <= bbox[1] and
                temp_bbox[2] >= bbox[2] and
                temp_bbox[3] >= bbox[3])
    else:
        return (temp_bbox[0] <= bbox[0] or
                temp_bbox[1] <= bbox[1] or
                temp_bbox[2] >= bbox[2] or
                temp_bbox[3] >= bbox[3])


def set_values_outside_bbox_to_zero(mask, bbox):
    """
    Set all the values of the mask outside the bounding box to zero
    :param mask: Mask as np.array uint8
    :param bbox: Bounding box as np.array in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    mask[:int(bbox[1]), :] = 0  # 0 to y_min-1
    mask[int(bbox[3]) + 1:, :] = 0  # y_max+1 to height
    mask[:, :int(bbox[0])] = 0  # 0 to x_min-1
    mask[:, int(bbox[2]) + 1:] = 0  # x_max+1 to width
    return mask


def dilate_pseudomasks(input_masks, path_bboxes, output_path):
    kernel = get_default_kernel()

    if len(input_masks) == 1:
        input_masks = glob.glob(os.path.expanduser(input_masks[0]))
        assert input_masks, "The input path(s) was not found"
    for path in tqdm.tqdm(input_masks):
        masks = np.load(path)['arr_0'].astype(np.uint8)  # H x W x n

        masks_height = masks.shape[0]
        masks_width = masks.shape[1]

        masks_id = os.path.basename(path)
        masks_id = os.path.splitext(masks_id)[0]

        bboxes = np.loadtxt(os.path.join(path_bboxes, f'{masks_id}.txt'), delimiter=" ", dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
        bboxes = bboxes[:, 1:]  # remove classes

        # Dilate the masks until they touch the edges of the bounding boxes
        for i in range(masks.shape[2]):
            if np.all((masks[:, :, i] == 0)):  # if empty mask
                continue
            abs_bbox = yolo_bbox_to_pascal_voc(bboxes[i], img_height=masks_height, img_width=masks_width)
            while not mask_touches_bbox(masks[:, :, i], abs_bbox, touches_all_edges=False):
                masks[:, :, i] = cv2.dilate(masks[:, :, i], kernel, iterations=1)
            masks[:, :, i] = set_values_outside_bbox_to_zero(masks[:, :, i], abs_bbox)

        # Save masks to file
        Path(output_path).mkdir(parents=True, exist_ok=True)
        save_masks(masks=masks, dest_folder=output_path, filename=f'{masks_id}.npz')