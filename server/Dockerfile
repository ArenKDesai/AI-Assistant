FROM ubuntu:24.04

COPY . /workspace
COPY ../comms/* /workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y wget \
unzip \
python3 \
python3-pip \
vim && \
pip3 install -r requirements.txt --break-system-packages
