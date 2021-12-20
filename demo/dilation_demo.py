import os.path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches

import utils.bbox_conversion
from pseudo_labeling.mask_processing import mask_touches_bbox, set_values_outside_bbox_to_zero, get_default_kernel


def plot_bboxes_yolo(image, bboxes, ax):
    # Yolo format: [x_center, y_center, width, height] (normalized)

    height = image.shape[0]
    width = image.shape[1]

    for bbox in bboxes:
        # Denormalization process
        x_cen = bbox[0] * width
        y_cen = bbox[1] * height
        w = bbox[2] * width
        h = bbox[3] * height

        # Obtain x_min and y_min
        x_min = x_cen - (w / 2)
        y_min = y_cen - (h / 2)

        # Draw rectangle
        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)


if __name__ == "__main__":
    img_id = "IMG_20210924_160704843"
    data_path = "./new_dataset/train"
    mask_path = "./pseudo_masks/new_dataset"

    image = cv2.imread(os.path.join(data_path, f'{img_id}.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    width = image.shape[1]

    bboxes = np.loadtxt(os.path.join(data_path, f'{img_id}.txt'), delimiter=" ", dtype=np.float32)
    if bboxes.ndim == 1:
        bboxes = np.expand_dims(bboxes, axis=0)
    bboxes = bboxes[:, 1:]

    masks = np.load(os.path.join(mask_path, f'{img_id}.npz'))['arr_0'].astype(np.uint8)

    index = 0
    mask = masks[:, :, index]
    bbox = bboxes[index]

    # Before dilation
    fig1, ax1 = plt.subplots()
    ax1.imshow(image)
    ax1.imshow(mask, alpha=0.5)
    plot_bboxes_yolo(image, [bbox], ax1)
    ax1.set_axis_off()
    plt.savefig('0_grape_none.png', bbox_inches='tight', pad_inches=0)

    # Dilation
    fig2, ax2 = plt.subplots()
    ax2.imshow(image)
    plot_bboxes_yolo(image, [bbox], ax2)
    abs_bbox = utils.bbox_conversion.yolo_bbox_to_pascal_voc(bbox, img_height=height, img_width=width)

    kernel = get_default_kernel()
    dilated_mask = mask.copy()
    if not np.all((dilated_mask == 0)):
        while not mask_touches_bbox(dilated_mask, abs_bbox, touches_all_edges=False):
            dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=1)
    set_values_outside_bbox_to_zero(dilated_mask, abs_bbox)
    dilated_mask = (dilated_mask - mask)

    ax2.imshow(dilated_mask+mask*3, alpha=0.5)
    ax2.set_axis_off()
    plt.savefig('0_grape_dilation.png', bbox_inches='tight', pad_inches=0)

    plt.show()
