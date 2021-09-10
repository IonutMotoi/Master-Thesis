import albumentations as A
import os
import cv2
import numpy as np

from utils.bbox_conversion import yolo_bboxes_to_albumentations, albumentations_bboxes_to_yolo


def save_augmented(dest_folder, img_id, image, bboxes, masks):
    # Save image
    image_path = os.path.join(dest_folder, f'{img_id}.jpg')
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Save bboxes
    bboxes_path = os.path.join(dest_folder, f'{img_id}.txt')
    np.savetxt(bboxes_path, bboxes, fmt="%i %.4f %.4f %.4f %.4f")

    # Save masks

    print(img_id, "saved successfully.")


def offline_augmentation(root, ids_txt, dest_folder, augmentations, augment_masks=True):
    ids_txt = os.path.join(root, ids_txt)

    with open(ids_txt, 'r') as f:
        # Read all lines in file
        lines = f.readlines()
        # Recover the items ids, removing the \n at the end
        ids = [line.rstrip() for line in lines]

    for img_id in ids:
        # Get image
        image_path = os.path.join(root, 'data', f'{img_id}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bboxes
        bboxes_path = os.path.join(root, 'data', f'{img_id}.txt')
        bboxes = np.loadtxt(bboxes_path, delimiter=" ", dtype=np.float32)
        head = bboxes[:, 0]
        bboxes = bboxes[:, 1:]
        bboxes = yolo_bboxes_to_albumentations(bboxes)

        if augment_masks:
            # Get masks
            masks_path = os.path.join(root, 'data', f'{img_id}.npz')
            masks = np.load(masks_path)['arr_0'].astype(np.uint8)
        else:
            masks = None

        # Augment image and annotations
        # TODO

        # Convert bboxes back to yolo format
        bboxes = albumentations_bboxes_to_yolo(bboxes)
        # Add object classes again
        bboxes = np.hstack((head[:, None], bboxes))

        # Save augmented image and annotations
        save_augmented(dest_folder, img_id, image, bboxes, masks)
        break


offline_augmentation(
        root="./wgisd",
        ids_txt="valid_split_masked.txt",
        dest_folder="./wgisd/augmented/valid_masked",
        augmentations=A.Compose([
            A.LongestMaxSize(max_size=1024),
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
        )], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels', 'bbox_ids'])),
        augment_masks=True)
