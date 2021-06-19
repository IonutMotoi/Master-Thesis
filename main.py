from data_loader import WGISDMaskedDataset
from mask_rcnn import get_mask_rcnn_model

import torch
from torch.utils.data import DataLoader
import torchvision


if __name__ == '__main__':
    # Train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("device = cuda")
    else:
        torch.device('cpu')
        print("device = cpu")

    # WGISD has only 2 classes: grapes and background
    num_classes = 2
    # Training and validation datasets
    dataset = WGISDMaskedDataset('./wgisd')
    dataset_test = WGISDMaskedDataset('./wgisd', source='test')

    # Training and validation data loaders
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    data_loader_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Get the model
    model = get_mask_rcnn_model(num_classes)

    # Move the model to the right device
    model.to(device)
