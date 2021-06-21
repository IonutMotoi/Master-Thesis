from data_loader import WGISDMaskedDataset
from mask_rcnn import get_mask_rcnn_model

from engine import train_one_epoch, evaluate
import utils

import torch
from torch.utils.data import DataLoader
import torchvision


if __name__ == '__main__':
    # Train on the GPU or on the CPU, if a GPU is not available
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     print("device = cuda")
    # else:
    device = torch.device('cpu')
    print("device = cpu")

    # WGISD has only 2 classes: grapes and background
    num_classes = 2
    # Training and validation datasets
    dataset = WGISDMaskedDataset('./wgisd')
    dataset_test = WGISDMaskedDataset('./wgisd', source='test')

    # Training and validation data loaders
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # Get the model
    model = get_mask_rcnn_model(num_classes)

    # Move the model to the right device
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # Update the learning rate
        lr_scheduler.step()
        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
