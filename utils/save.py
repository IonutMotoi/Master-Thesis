import os
from pathlib import Path

import cv2
import numpy as np


def save_image_and_labels(dest_folder, img_id, image, class_labels, bboxes, masks, img_format="RGB"):
    # Create destination folder
    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    # Save image
    image_path = os.path.join(dest_folder, f'{img_id}.jpg')
    assert img_format in ["RGB", "BGR"]
    if img_format == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)

    # Add object classes to bounding boxes
    bboxes = np.hstack((np.array(class_labels)[:, None], bboxes))

    # Save bboxes
    bboxes_path = os.path.join(dest_folder, f'{img_id}.txt')
    np.savetxt(bboxes_path, bboxes, fmt="%i %.4f %.4f %.4f %.4f")

    if masks is not None:
        # Save masks
        masks_path = os.path.join(dest_folder, f'{img_id}.npz')
        np.savez_compressed(masks_path, masks)


def save_masks(masks, dest_folder, filename):
    masks_path = os.path.join(dest_folder, filename)
    np.savez_compressed(masks_path, masks)
