<img src="mewc_logo_hex.png" alt="MEWC Hex Sticker" width="200" align="right"/>

# mewc-flow

This repository contains the Dockerfile and docker-compose.yml files used to build the `mewc-flow` Docker image. The `mewc-flow` Docker image serves as the base image for the `mewc-train` and `mewc-predict` Docker images, which are used for Efficient Net v2 training and prediction for wildlife camera trap images.

- [mewc-train Docker Image Repository](https://github.com/zaandahl/mewc-train)
- [mewc-predict Docker Image Repository](https://github.com/zaandahl/mewc-predict)

The `mewc-flow` image is built on top of the latest stable Python image and includes additional dependencies required by Efficient Net v2 models.

## Efficient Net v2

Efficient Net v2 is a scalable neural network architecture designed for efficiency and accuracy, which is particularly suited to the classification of wildlife images. More information about Efficient Net v2 can be found in its official documentation:

- [Efficient Net v2 Documentation](https://github.com/google/automl/tree/master/efficientnetv2)

## Building the Docker Image

This repository uses `docker-compose` to build the Docker image. To build the `mewc-flow` Docker image, clone this repository and use the `docker-compose` command:

```bash
git clone https://github.com/zaandahl/mewc-flow.git
cd mewc-flow
docker-compose up --build
```

This will create a Docker image named `zaandahl/mewc-flow`.

## Docker Image Contents
The Docker image contains:

- Python environment with the requirements from requirements.txt file installed.
- Necessary utilities (ffmpeg, libsm6, libxext6, git, wget) installed.
- Source code copied into the /code directory in the container.

For detailed information about the image contents, please refer to the Dockerfile in this repository.

## GitHub Actions and DockerHub
This project uses GitHub Actions to automate the build process and push the Docker image to DockerHub. You can find the image at:

- [zaandahl/mewc-flow DockerHub Repository](https://hub.docker.com/repository/docker/zaandahl/mewc-flow)

