FROM ubuntu:24.04

COPY ../ .

RUN apt-get update && apt-get install -y curl \
    python3 \
    python3-pip \
    vim \
    && pip install -r server-requirements.txt --break-system-packages

CMD ["python3", "server.py"]
