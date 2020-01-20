FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y --allow-unauthenticated \
    build-essential \
    python3-dev \
    python3-pip 

COPY pytorch_organisation_lang /pytorch_organisation_lang

WORKDIR /pytorch_organisation_lang

RUN pip3 install --upgrade pip
RUN pip3 install -r /pytorch_organisation_lang/requierements.txt
