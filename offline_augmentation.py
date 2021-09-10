import albumentations as A
import os
import cv2
import numpy as np

from utils.bbox_conversion import yolo_bboxes_to_albumentations, albumentations_bboxes_to_yolo


def save_augmented(dest_folder, img_id, image, bboxes, masks, count, tot_imgs):
    # Save image
    image_path = os.path.join(dest_folder, f'{img_id}.jpg')
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Save bboxes
    bboxes_path = os.path.join(dest_folder, f'{img_id}.txt')
    np.savetxt(bboxes_path, bboxes, fmt="%i %.4f %.4f %.4f %.4f")

    if masks is not None:
        # Save masks
        masks_path = os.path.join(dest_folder, f'{img_id}.npz')
        np.savez_compressed(masks_path, masks)

    print(f"[{count:2}/{tot_imgs:2}] {img_id} saved successfully.")


def offline_augmentation(ids_txt, data_folder, dest_folder, augmentations, augment_masks):
    with open(ids_txt, 'r') as f:
        # Read all lines in file
        lines = f.readlines()
        # Recover the items ids, removing the \n at the end
        ids = [line.rstrip() for line in lines]
    tot_imgs = len(ids)

    for count, img_id in enumerate(ids, start=1):
        # Get image
        image_path = os.path.join(data_folder, f'{img_id}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bboxes and class labels
        bboxes_path = os.path.join(data_folder, f'{img_id}.txt')
        bboxes = np.loadtxt(bboxes_path, delimiter=" ", dtype=np.float32)
        class_labels = bboxes[:, 0].astype(np.uint8)
        bboxes = bboxes[:, 1:]
        bboxes = yolo_bboxes_to_albumentations(bboxes)

        if augment_masks:
            # Get masks
            masks_path = os.path.join(data_folder, f'{img_id}.npz')
            masks_arr = np.load(masks_path)['arr_0'].astype(np.uint8)
            masks = [masks_arr[:, :, i] for i in range(masks_arr.shape[-1])]
        else:
            masks = None

        # Augment image and annotations
        augmented = augmentations(
            image=image,
            bboxes=bboxes,
            masks=masks,
            class_labels=class_labels
        )
        image = augmented["image"]
        bboxes = augmented["bboxes"]
        masks = augmented["masks"]
        class_labels = augmented["class_labels"]

        # Convert bboxes back to yolo format
        bboxes = albumentations_bboxes_to_yolo(bboxes)
        # Add object classes again
        bboxes = np.hstack((np.array(class_labels)[:, None], bboxes))

        masks = np.array(masks).transpose((1, 2, 0))  # n x H x W -> H x W x n

        # Save augmented image and annotations
        save_augmented(dest_folder, img_id, image, bboxes, masks, count, tot_imgs)


offline_augmentation(
    ids_txt="./wgisd/valid_split_masked.txt",
    data_folder="./wgisd/data",
    dest_folder="./wgisd/augmented/valid_masked",
    augmentations=A.Compose([
        A.LongestMaxSize(max_size=1024),
        A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels'])),
    augment_masks=True)
