FROM ubuntu:24.04

COPY . /workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y curl \
    python3 \
    python3-pip \
    vim \
    && pip install -r src/server/server-requirements.txt --break-system-packages

# The user will need to log into the HuggingFace CLI before starting server.py
# Build this container with $ docker build . -f server.Dockerfile -t aia-server
# NOTE: This may take 35+ minutes.
# Then, enter the container with $ docker run -it aia-server
# Finally, run server.py in /workspace/src/server
