import os.path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches
from skimage.segmentation import slic, mark_boundaries

import utils.bbox_conversion
from pseudo_labeling.mask_processing import mask_touches_bbox, set_values_outside_bbox_to_zero, get_default_kernel
import time

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
    # img_id = "IMG_20210924_171537076"
    img_id = "IMG_20210924_161822966_HDR"
    # img_id = "IMG_20210924_132053023"
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

    mask = masks[:, :, 0]
    bbox = bboxes[0]

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig1, ax1 = plt.subplots(1, 1)

    # Before processing
    ax1.imshow(image)
    ax1.imshow(mask, alpha=0.5)
    ax1.set_axis_off()
    # plot_bboxes_yolo(image, [bbox], ax1)

    # Clustering
    # ax2.imshow(image)

    image_resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR INTER_AREA
    mask_resized = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    t0 = time.time()
    segments = slic(image_resized, slic_zero=False, max_iter=10, n_segments=2000, compactness=0.1,
                    start_label=1, convert2lab=True, sigma=0)
    print(time.time() - t0)

    fig2, ax2 = plt.subplots(1, 1)
    ax2.imshow(mark_boundaries(image_resized, segments, outline_color=(255.0/255, 20.0/255, 147.0/255)))
    # ax2.imshow(image_resized)
    # ax2.imshow(mask_resized, alpha=0.5)
    ax2.set_axis_off()
    plt.savefig(f'{img_id}.png', bbox_inches='tight', pad_inches=0, dpi=300)

    t0 = time.time()
    print(len(np.unique(segments)[1:]))

    mask_resized2 = mask_resized.copy()
    mask_resized3 = mask_resized.copy()

    # for n, i in enumerate(np.unique(segments)[1:]):
    #     # print(f"{n+1}/{len(np.unique(segments)[1:])}")
    #     cluster = segments == i
    #
    #     intersection_area = np.sum((cluster * mask_resized) > 0, axis=(0, 1))
    #     cluster_area = np.sum(cluster > 0, axis=(0, 1))
    #
    #     if intersection_area / cluster_area > 0.1:
    #         mask_resized = ((cluster + mask_resized) > 0).astype(np.uint8)
    #     # if intersection_area / cluster_area < 0.3:
    #     #     mask_resized = (mask_resized - cluster*mask_resized).astype(np.uint8)
    # print(time.time() - t0)
    # mask_resized = cv2.resize(mask_resized, (width, height), interpolation=cv2.INTER_LINEAR)
    #
    # fig3, ax3 = plt.subplots(1, 1)
    # ax3.imshow(image)
    # ax3.imshow(mask_resized, alpha=0.5)
    # ax3.set_axis_off()
    # plt.savefig(f'{img_id}_slic_01.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # for n, i in enumerate(np.unique(segments)[1:]):
    #     # print(f"{n+1}/{len(np.unique(segments)[1:])}")
    #     cluster = segments == i
    #
    #     intersection_area = np.sum((cluster * mask_resized2) > 0, axis=(0, 1))
    #     cluster_area = np.sum(cluster > 0, axis=(0, 1))
    #
    #     if intersection_area / cluster_area > 0.7:
    #         mask_resized2 = ((cluster + mask_resized2) > 0).astype(np.uint8)
    #     # if intersection_area / cluster_area < 0.3:
    #     #     mask_resized2 = (mask_resized2 - cluster*mask_resized2).astype(np.uint8)
    # print(time.time() - t0)
    # mask_resized2 = cv2.resize(mask_resized2, (width, height), interpolation=cv2.INTER_LINEAR)
    #
    # fig4, ax4 = plt.subplots(1, 1)
    # ax4.imshow(image)
    # ax4.imshow(mask_resized2, alpha=0.5)
    # ax4.set_axis_off()
    # plt.savefig(f'{img_id}_slic_07.png', bbox_inches='tight', pad_inches=0, dpi=300)

    for n, i in enumerate(np.unique(segments)[1:]):
        # print(f"{n+1}/{len(np.unique(segments)[1:])}")
        cluster = segments == i

        intersection_area = np.sum((cluster * mask_resized3) > 0, axis=(0, 1))
        cluster_area = np.sum(cluster > 0, axis=(0, 1))

        if intersection_area / cluster_area > 0.7:
            mask_resized3 = ((cluster + mask_resized3) > 0).astype(np.uint8)
        if intersection_area / cluster_area < 0.3:
            mask_resized3 = (mask_resized3 - cluster*mask_resized3).astype(np.uint8)
    print(time.time() - t0)
    mask_resized3 = cv2.resize(mask_resized3, (width, height), interpolation=cv2.INTER_LINEAR)

    fig5, ax5 = plt.subplots(1, 1)
    ax5.imshow(image)
    ax5.imshow(mask_resized3, alpha=0.5)
    ax5.set_axis_off()
    plt.savefig(f'{img_id}_slic_07_03.png', bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()
