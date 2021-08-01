import pycocotools
import os
import cv2
import numpy as np

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog


# Dataset (with masks)
def get_wgisd_dicts(root, source):

  # Load the dataset subset defined by source
  assert source in ('train', 'valid', 'test'), \
        'source should be "train", "valid" or "test"'

  if source == 'train':
    source_path = os.path.join(root, 'train_split_masked.txt')
  elif source == 'valid':
    source_path = os.path.join(root, 'valid_split_masked.txt')
  else:
    source_path = os.path.join(root, 'test_masked.txt')

  with open(source_path, 'r') as fp:
    # Read all lines in file
    lines = fp.readlines()
    # Recover the items ids, removing the \n at the end
    ids = [l.rstrip() for l in lines]

  imgs = [os.path.join(root, 'data', f'{id}.jpg') for id in ids]
  masks = [os.path.join(root, 'data', f'{id}.npz') for id in ids]
  boxes = [os.path.join(root, 'data', f'{id}.txt') for id in ids]

  dataset_dicts = []
  for id in ids:
    record = {}
    
    filename = os.path.join(root, 'data', f'{id}.jpg')
    height, width = cv2.imread(filename).shape[:2]

    record["file_name"] = filename
    record["image_id"] = id
    record["height"] = height
    record["width"] = width
  
    box_path = os.path.join(root, 'data', f'{id}.txt')
    mask_path = os.path.join(root, 'data', f'{id}.npz')

    wgisd_masks = np.load(mask_path)['arr_0'].astype(np.uint8)
    num_objs = wgisd_masks.shape[2]

    boxes_text = np.loadtxt(box_path, delimiter=" ", dtype=np.float32)
    wgisd_boxes = boxes_text[:, 1:]
    assert (wgisd_boxes.shape[0] == num_objs)    

    objs = []
    for i in range(num_objs):
      box = wgisd_boxes[i]
      mask = wgisd_masks[:,:,i]

      # Boxes (x0, y0, w, h) in range [0, 1]
      # They are relative to the size of the image
      # Convert to (x0, y0, x1, y1) in absolute floating points coordinates
      x1 = box[0] - box[2] / 2
      x2 = box[0] + box[2] / 2
      y1 = box[1] - box[3] / 2
      y2 = box[1] + box[3] / 2
      box = [x1 * width, y1 * height, x2 * width, y2 * height]
      
      # Convert a uint8 mask of 0s and 1s into dict 
      # with keys “size” and “counts”
      mask = pycocotools.mask.encode(np.asarray(mask, order="F"))
      
      obj = {
          "bbox": box,
          "bbox_mode": BoxMode.XYXY_ABS,
          "segmentation": mask,
          "category_id": 0,
      }
      objs.append(obj)

    record["annotations"] = objs
    dataset_dicts.append(record)
  return dataset_dicts

def setup_wgisd():
  data_path = "/datasets/wgisd"

  for d in ["train", "valid", "test"]:
    dataset_name = "wgisd_" + d
    if dataset_name in DatasetCatalog.list():
      DatasetCatalog.remove(dataset_name)

    DatasetCatalog.register(dataset_name, lambda d=d: get_wgisd_dicts(data_path, d))
    MetadataCatalog.get(dataset_name).set(thing_classes=["grapes"])

  MetadataCatalog.get("wgisd_valid").set(evaluator_type = "coco")
  MetadataCatalog.get("wgisd_test").set(evaluator_type = "coco")