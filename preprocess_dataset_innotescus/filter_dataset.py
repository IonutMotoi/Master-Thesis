from pathlib import Path
import shutil

if __name__ == '__main__':
    # Use masked images as validation / test and images with only bboxes as train

    path_dataset = Path("./dataset")
    path_train = Path("./new_dataset", "train")
    path_train.mkdir(parents=True, exist_ok=True)
    path_validation = Path("./new_dataset", "validation")
    path_validation.mkdir(parents=True, exist_ok=True)
    path_test = Path("./new_dataset", "test")
    path_test.mkdir(parents=True, exist_ok=True)

    ids = [file.stem for file in path_dataset.glob("*")]
    print("Total number of images:", len(ids))

    ids_with_masks = [file.stem for file in Path.cwd().glob("annotations/*.png")]
    print("Images with mask annotations:", len(ids_with_masks))

    ids_without_masks = [id for id in ids if id not in ids_with_masks]
    print("Images with only bounding box annotations:", len(ids_without_masks))

    valid_split = len(ids_with_masks) // 2
    print("Train split:", len(ids_without_masks))
    print("Validation split:", valid_split)
    print("Test split:", len(ids_with_masks) - valid_split)

    for i, id_ in enumerate(sorted(ids_with_masks)):
        file = path_dataset.joinpath(id_)
        file = file.with_suffix(".jpg")
        if i < valid_split:
            shutil.copy(file, path_validation)
        else:
            shutil.copy(file, path_test)

    for id_ in ids_without_masks:
        file = path_dataset.joinpath(id_)
        file = file.with_suffix(".jpg")
        shutil.copy(file, path_train)

    print("Done copy of", len(ids), "images")
