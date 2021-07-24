import os
import torch, torchvision
import numpy as np
import pycocotools

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.structures import BoxMode
