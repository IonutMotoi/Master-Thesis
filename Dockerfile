FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0

RUN pip install notebook

COPY wgisd /datasets/wgisd