import os
import torch, torchvision
import numpy as np
import pycocotools

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup


def main(args):
    print("Hello")

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
