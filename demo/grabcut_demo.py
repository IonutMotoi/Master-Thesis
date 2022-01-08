import os.path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches
from skimage.segmentation import slic, mark_boundaries

import utils.bbox_conversion
from pseudo_labeling.mask_processing import mask_touches_bbox, set_values_outside_bbox_to_zero, get_default_kernel, dilate_pseudomasks
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
    # img_id = "IMG_20210924_131131409"
    img_id = "IMG_20210924_161822966_HDR"
    # img_id = "IMG_20210924_160155838"
    # img_id = "IMG_20210924_131835597"
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

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig1, ax1 = plt.subplots()

    # Before processing
    ax1.imshow(image)
    ax1.imshow(mask, alpha=0.5)
    plot_bboxes_yolo(image, [bbox], ax1)
    ax1.set_axis_off()
    plt.savefig('grabcut_1_before.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # Grabcut
    fig2, ax2 = plt.subplots()
    ax2.imshow(image)
    plot_bboxes_yolo(image, [bbox], ax2)
    abs_bbox = utils.bbox_conversion.yolo_bbox_to_pascal_voc(bbox, img_height=height, img_width=width)

    # allocate memory for two arrays that the GrabCut algorithm internally
    # uses when segmenting the foreground from the background
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    # apply GrabCut using the mask initialization method
    # mask_dilated = dilate_pseudomasks(np.array([mask.copy()]).transpose((1, 2, 0)), [bbox])
    # mask_dilated = mask_dilated.squeeze()
    # mask_dilated = mask_dilated - mask
    # mask[mask_dilated > 0] = cv2.GC_PR_FGD
    iters = int(min(abs_bbox[2] - abs_bbox[0], abs_bbox[3] - abs_bbox[1]) / 40)
    print(iters)
    eroded_mask = cv2.erode(mask, get_default_kernel(), iterations=iters)
    dilated_mask = cv2.dilate(mask, get_default_kernel(), iterations=iters)
    new_mask = mask.copy()
    new_mask[dilated_mask > 0] = cv2.GC_PR_BGD
    new_mask[mask > 0] = cv2.GC_PR_FGD
    new_mask[eroded_mask > 0] = cv2.GC_FGD
    set_values_outside_bbox_to_zero(new_mask, abs_bbox)
    print(np.unique(new_mask))
    ax2.imshow(new_mask, alpha=0.5)
    ax2.set_axis_off()
    plt.savefig('grabcut_2_mask.png', bbox_inches='tight', pad_inches=0, dpi=300)

    start = time.time()
    (mask_grabcut, bgModel, fgModel) = cv2.grabCut(image, new_mask.copy(), None, bgModel,
                                           fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    end = time.time()
    print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))
    mask_grabcut = np.where((mask_grabcut == cv2.GC_BGD) | (mask_grabcut == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    fig3, ax3 = plt.subplots()
    ax3.imshow(image)
    ax3.imshow(mask_grabcut, alpha=0.5)
    ax3.set_axis_off()
    plt.savefig('grabcut_3_after.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # ax3.imshow((mask_grabcut == cv2.GC_BGD) | (mask_grabcut == cv2.GC_PR_BGD), alpha=0.5)
    # ax3.imshow((mask_grabcut == cv2.GC_FGD), alpha=0.5)
    # ax3.imshow((mask_grabcut == cv2.GC_PR_FGD), alpha=0.5)
    # ax3.imshow(mask_grabcut, alpha=0.5)

    mask_grabcut = cv2.medianBlur(mask_grabcut, 25)
    fig4, ax4 = plt.subplots()
    ax4.imshow(image)
    ax4.imshow(mask_grabcut, alpha=0.5)
    ax4.set_axis_off()
    plt.savefig('grabcut_4_medianblur.png', bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()
