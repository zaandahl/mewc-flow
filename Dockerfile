# set base image (host OS)
# Digest from TensorFlow nightly-gpu on 2024-07-25 
FROM tensorflow/tensorflow@sha256:c73a8dafeb4254896fd9fc8db7f5e748a6bbb4242937a7a14c9e09feb49cdcdc
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

