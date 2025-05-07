# Base image: CUDA 11.8 compatible with PyTorch 2.x
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

LABEL maintainer="Your Name <youremail@example.com>"
LABEL description="Docker container for VGGT (new workflow) and COLMAP conversion."

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV VENV_PATH=/opt/venv_vggt
# For Hugging Face model caching
ENV HF_HOME=/opt/hf_cache
ENV HF_HUB_CACHE=${HF_HOME}/hub
ENV TRANSFORMERS_CACHE=${HF_HOME}/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    unzip \
    ca-certificates \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    dos2unix \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python${PYTHON_VERSION} -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"
RUN pip install --upgrade pip

# Install PyTorch (>=2.0 as per requirements.txt) with CUDA 11.8
# Using torch 2.1.0 as an example, check requirements.txt for latest compatible
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Set up the vggt application
WORKDIR /app

# Clone VGGT repository
RUN git clone https://github.com/facebookresearch/vggt.git
WORKDIR /app/vggt

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt
# Add other common dependencies that vggt_to_colmap or our script might need
RUN pip install opencv-python scipy matplotlib scikit-image einops

# Install the vggt package itself from the cloned repo to make 'import vggt' work
RUN echo "Installing VGGT package from local source (/app/vggt)..."
RUN pip install .

# Verify installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
RUN python -c "import PIL; import huggingface_hub; print('Pillow and huggingface_hub imported')"
RUN python -c "import vggt; print('VGGT package imported successfully')"


# Copy our custom inference script and the main pipeline script
# These should be in the same directory as the Dockerfile when building
COPY run_vggt_inference.py /app/run_vggt_inference.py
COPY run_vggt_colmap.sh /app/run_vggt_colmap.sh

RUN dos2unix /app/run_vggt_inference.py && chmod +x /app/run_vggt_inference.py
RUN dos2unix /app/run_vggt_colmap.sh && chmod +x /app/run_vggt_colmap.sh

# Create Hugging Face cache directory and set permissions
# This directory will be used by huggingface_hub to download models
RUN mkdir -p ${HF_HUB_CACHE} && \
    chmod -R 777 ${HF_HOME}

# Set a working directory for processing data (where volumes will be mounted)
WORKDIR /workspace

# Entrypoint to our main pipeline script
ENTRYPOINT ["/app/run_vggt_colmap.sh"]