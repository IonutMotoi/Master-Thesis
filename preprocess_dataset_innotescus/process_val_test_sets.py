from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def save_masks(masks_list, image_id, dest_folder):
    dest_folder = Path("./datasets", "new_dataset") / dest_folder
    masks = np.stack(masks_list, axis=2)  # HxWxN
    print(" Shape of masks array:", masks.shape)
    masks_path = dest_folder / f'{image_id}.npz'
    np.savez_compressed(masks_path, masks)


if __name__ == '__main__':
    with open('./datasets/TGIS/classes.txt', "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    ids_validation = [file.stem for file in Path.cwd().glob("datasets/new_dataset/validation/*.jpg")]
    ids_test = [file.stem for file in Path.cwd().glob("datasets/new_dataset/test/*.jpg")]

    instances_list = []
    curr_mask_id = ""
    for line in tqdm(lines):
        if line.split()[0] == "file:":  # file id
            if instances_list:  # save the masks of the previous file
                curr_mask_id = curr_mask_id.split(".")[0]
                if curr_mask_id in ids_validation:
                    save_masks(instances_list, curr_mask_id, "validation")
                elif curr_mask_id in ids_test:
                    save_masks(instances_list, curr_mask_id, "test")
                instances_list = []
            curr_mask_id = line.split()[1]
            path = Path("./datasets/TGIS") / curr_mask_id
            mask = np.array(Image.open(path))
        else:  # instances
            instance_id, category = line.split(maxsplit=1)
            if "pizzutello" in category:
                instances_list.append(np.where(mask == int(instance_id), 1, 0))
    # save the masks of the last file
    curr_mask_id = curr_mask_id.split(".")[0]
    if curr_mask_id in ids_validation:
        save_masks(instances_list, curr_mask_id, "validation")
    elif curr_mask_id in ids_test:
        save_masks(instances_list, curr_mask_id, "test")
