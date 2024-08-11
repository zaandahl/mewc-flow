# set base image (host OS)
FROM tensorflow/tensorflow:2.16.1-gpu
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    numactl \
    nvidia-modprobe \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy the script to modify NUMA nodes
COPY modify_numa.sh /usr/local/bin/modify_numa.sh

# Make the script executable
RUN chmod +x /usr/local/bin/modify_numa.sh

# Run the script during the build process
RUN /usr/local/bin/modify_numa.sh

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# Set environment variables for CUDA and cuDNN
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# install dependencies
RUN pip install -r requirements.txt

# Install JAX and JAXLIB with CUDA 12 support
RUN pip install --upgrade jax==0.4.28 jaxlib==0.4.28+cuda12_pip -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# copy code
COPY src/ .
