FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0

RUN pip install notebook

COPY wgisd /thesis/wgisd
COPY detectron2_baseline /thesis/detectron2_baseline