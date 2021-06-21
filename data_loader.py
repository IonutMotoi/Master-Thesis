import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class WGISDMaskedDataset(Dataset):
    def __init__(self, root, transforms=None, source='train'):
        self.root = root
        self.transforms = transforms

        # Load the dataset subset defined by source
        if source not in ('train', 'test'):
            print('source should by "train" or "test"')
            return None

        source_path = os.path.join(root, f'{source}_masked.txt')
        with open(source_path, 'r') as fp:
            # Read all lines in file
            lines = fp.readlines()
            # Recover the items ids, removing the \n at the end
            ids = [l.rstrip() for l in lines]

        self.imgs = [os.path.join(root, 'data', f'{id}.jpg') for id in ids]
        self.masks = [os.path.join(root, 'data', f'{id}.npz') for id in ids]
        self.boxes = [os.path.join(root, 'data', f'{id}.txt') for id in ids]

    def __getitem__(self, idx):
        # Load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        box_path = self.boxes[idx]

        img = Image.open(img_path).convert("RGB")

        # From TorchVision documentation:
        #
        # The models expect a list of Tensor[C, H, W], in the range 0-1.
        # The models internally resize the images so that they have a minimum
        # size of 800. This option can be changed by passing the option min_size
        # to the constructor of the models.

        if self.transforms is None:
            img = np.array(img)
            # Normalize
            img = (img - img.min()) / np.max([img.max() - img.min(), 1])
            # Move the channels axe to the first position, getting C, H, W instead H, W, C
            img = np.moveaxis(img, -1, 0)
            img = torch.as_tensor(img, dtype=torch.float32)
        else:
            img = np.array(img)
            img = self.transforms(torch.as_tensor(img, dtype=torch.uint8))
            img = np.array(img)
            # Normalize
            img = (img - img.min()) / np.max([img.max() - img.min(), 1])
            img = np.moveaxis(img, -1, 0)
            img = torch.as_tensor(img, dtype=torch.float32)

        # Loading masks:
        #
        # As seen in WGISD (README.md):
        #
        # After assigning the NumPy array to a variable M, the mask for the
        # i-th grape cluster can be found in M[:,:,i]. The i-th mask corresponds
        # to the i-th line in the bounding boxes file.
        #
        # According to Mask RCNN documentation in Torchvision:
        #
        # During training, the model expects both the input tensors, as well as
        # a targets (list of dictionary), containing:
        # (...)
        # masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each
        # instance
        #
        # WGISD provides [H, W, N] masks, but Torchvision asks for [N, H, W].
        # Let's employ NumPy moveaxis.
        wgisd_masks = np.load(mask_path)['arr_0'].astype(np.uint8)
        masks = np.moveaxis(wgisd_masks, -1, 0)

        num_objs = masks.shape[0]
        all_text = np.loadtxt(box_path, delimiter=" ", dtype=np.float32)
        wgisd_boxes = all_text[:, 1:]
        assert (wgisd_boxes.shape[0] == num_objs)

        # According to WGISD:
        #
        # These text files follows the "YOLO format"
        #
        # CLASS CX CY W H
        #
        # CLASS is an integer defining the object class – the dataset presents
        # only the grape class that is numbered 0, so every line starts with
        # this "class zero" indicator.
        # The center of the bounding box is the point (c_x, c_y), represented
        # as float values because this format normalizes the coordinates by
        # the image dimensions.
        # To get the absolute position, use (2048 c_x, 1365 c_y).
        # The bounding box dimensions are given by W and H, also normalized by
        # the image size.
        #
        # Torchvision's Mask R-CNN expects absolute coordinates.
        _, height, width = img.shape

        boxes = []
        for box in wgisd_boxes:
            x1 = box[0] - box[2] / 2
            x2 = box[0] + box[2] / 2
            y1 = box[1] - box[3] / 2
            y2 = box[1] + box[3] / 2
            boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])

        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # There is only one class -> grapes
        # IMPORTANT: Torchvision considers 0 as background.
        # So, let's make grapes as class 1
        labels = torch.ones((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd" : iscrowd
        }

        return img, target

    def __len__(self):
        return len(self.imgs)
