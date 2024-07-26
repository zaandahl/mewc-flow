# set base image (host OS)
# Digest from TensorFlow nightly-gpu on 2024-07-25 
FROM tensorflow/tensorflow@sha256:c73a8dafeb42
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy code
COPY src/ .

