import cv2
from detectron2.data import detection_utils
from detectron2.utils.logger import setup_logger
from matplotlib import pyplot as plt
import albumentations as A
import detectron2.data.transforms as T


from utils.albumentations_mapper import get_augmentations
from utils.inference_setup import get_parser, setup

if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    image_path = "./wgisd/data/CDY_2031.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    augmentations = A.Compose(get_augmentations(cfg))

    # Define transforms (Detectron2)
    transforms = detection_utils.build_augmentation(cfg, is_train=True)
    transforms.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    transforms = T.AugmentationList(transforms)

    for i in range(24):
        augmented = augmentations(image=image)
        augmented_image = augmented["image"]

        aug_input = T.AugInput(augmented_image)
        applied_transforms = transforms(aug_input)
        augmented_image = aug_input.image

        cv2.imwrite(f'augmented_image_{i}.jpg', cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

    plt.imshow(image)
    plt.show()
