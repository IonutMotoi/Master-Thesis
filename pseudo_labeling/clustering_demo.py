import os.path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches
from skimage.color import rgb2gray
from skimage.filters import sobel

import utils.bbox_conversion
from pseudo_labeling.mask_processing import mask_touches_bbox, set_values_outside_bbox_to_zero, get_default_kernel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.util import img_as_float


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
    # img_id = "IMG_20210924_131131409"
    # img_id = "IMG_20210924_112427127"
    img_id = "IMG_20210924_132053023"
    data_path = "./new_dataset/train"
    mask_path = "./pseudo_labels"

    image = cv2.imread(os.path.join(data_path, f'{img_id}.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    width = image.shape[1]

    bboxes = np.loadtxt(os.path.join(data_path, f'{img_id}.txt'), delimiter=" ", dtype=np.float32)
    if bboxes.ndim == 1:
        bboxes = np.expand_dims(bboxes, axis=0)
    bboxes = bboxes[:, 1:]

    masks = np.load(os.path.join(mask_path, f'{img_id}.npz'))['arr_0'].astype(np.uint8)

    mask = masks[:, :, 0]
    bbox = bboxes[0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Before processing
    ax1.imshow(image)
    ax1.imshow(mask, alpha=0.5)
    plot_bboxes_yolo(image, [bbox], ax1)

    # Clustering
    ax2.imshow(image)
    # low_res_image = image[::8,::8]
    plot_bboxes_yolo(image, [bbox], ax2)
    abs_bbox = utils.bbox_conversion.yolo_bbox_to_pascal_voc(bbox, img_height=height, img_width=width)
    slic_mask = np.ones((image.shape[0], image.shape[1]))
    set_values_outside_bbox_to_zero(slic_mask, abs_bbox)

    segments = slic(image, slic_zero=False, n_segments=1000, compactness=20, start_label=1, convert2lab=True, sigma=0)
    ax2.imshow(segments)
    ax2.imshow(mask, alpha=0.5)

    for n, i in enumerate(np.unique(segments)[1:]):
        print(f"{n+1}/{len(np.unique(segments))-1}")
        cluster = segments == i

        intersection_area = np.sum((cluster * mask) > 0, axis=(0, 1))
        cluster_area = np.sum(cluster > 0, axis=(0, 1))

        if intersection_area / cluster_area > 0.7:
            mask = ((cluster + mask) > 0).astype(np.uint8)
        if intersection_area / cluster_area < 0.3:
            mask = (mask - ((cluster * mask) > 0)).astype(np.uint8)

    ax3.imshow(image)
    ax3.imshow(mask, alpha=0.5)

    plt.show()
