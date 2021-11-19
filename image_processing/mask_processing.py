import numpy as np


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
