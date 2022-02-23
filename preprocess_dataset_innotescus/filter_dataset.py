from pathlib import Path
import shutil

if __name__ == '__main__':
    # Split dataset from Innotescus in train/val/test

    # Sources
    path_TGD = Path("./datasets", "TGD")  # contains yolo bounding boxes (txt)
    path_TGIS = Path("./datasets", "TGIS")  # contains instance annotations (png)
    path_TG = Path("./datasets", "TG")  # contains all images (jpg)
    # Destinations
    path_train = Path("./datasets", "new_dataset", "train")
    path_train.mkdir(parents=True, exist_ok=True)
    path_validation = Path("./datasets", "new_dataset", "validation")
    path_validation.mkdir(parents=True, exist_ok=True)
    path_test = Path("./datasets", "new_dataset", "test")
    path_test.mkdir(parents=True, exist_ok=True)

    ids_tgis = [file.stem for file in path_TGIS.glob("*.png")]
    print("Total number of images in TIS:", len(ids_tgis))

    ids_tgd = [file.stem for file in path_TGD.glob("*.txt")]
    print("Total number of images in TGD:", len(ids_tgd))

    for i, id_ in enumerate(ids_tgis):
        image = path_TG.joinpath(id_)
        image = image.with_suffix(".jpg")
        if i < 20:
            shutil.copy(image, path_test)
        elif i < 40:
            shutil.copy(image, path_validation)
        else:
            shutil.copy(image, path_train)

    for id_ in ids_tgd:
        if id_ not in ids_tgis:
            image = path_TG.joinpath(id_)
            image = image.with_suffix(".jpg")
            shutil.copy(image, path_train)

    print("Done")
