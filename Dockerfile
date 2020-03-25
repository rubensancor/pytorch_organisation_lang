FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y --allow-unauthenticated \
    build-essential \
    python3-dev \
    python3-pip 

COPY requierements.txt /

WORKDIR /

RUN pip3 install --upgrade pip
RUN pip3 install -r /requierements.txt
