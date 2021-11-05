FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN pip install opencv-python-headless

RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

RUN pip install notebook

RUN pip install wandb

RUN pip install -U albumentations

COPY detectron2_baseline /thesis/detectron2_baseline
COPY wgisd /thesis/wgisd