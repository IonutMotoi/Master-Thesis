from pathlib import Path

from tqdm import tqdm

if __name__ == '__main__':
    path_annotations = Path("./datasets", "TGD")
    path_dataset = Path("./datasets", "new_dataset", "train")

    ids = [file.stem for file in path_dataset.glob("*.jpg")]

    for id_ in tqdm(ids):
        with open(f'./datasets/TGD/{id_}.txt', "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]

        with open(path_dataset / (id_ + ".txt"), "w") as f:
            for line in lines:
                category, bbox = line.split(maxsplit=1)
                if int(category) == 5:
                    f.write("0 " + bbox + "\n")
