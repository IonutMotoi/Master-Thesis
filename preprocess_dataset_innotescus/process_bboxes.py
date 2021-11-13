from pathlib import Path

from tqdm import tqdm

if __name__ == '__main__':
    path_dataset = Path("./dataset")
    path_train = Path("./new_dataset", "train")

    ids = [file.stem for file in path_dataset.glob("*")]
    ids_with_masks = [file.stem for file in Path.cwd().glob("annotations/*.png")]
    ids_train = [id_ for id_ in ids if id_ not in ids_with_masks]

    for id_ in tqdm(ids_train):
        with open(f'./annotations/{id_}.txt', "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]

        with open(path_train / (id_ + ".txt"), "w") as f:
            for line in lines:
                category, bbox = line.split(maxsplit=1)
                if int(category) in [4, 5]:
                    f.write("0 " + bbox + "\n")
