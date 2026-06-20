FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    wandb \
    scipy

# SkyPilot's Kubernetes runtime sets up SSH between the head and worker pods and
# syncs files with rsync, so the image needs an SSH server and rsync present.
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server rsync \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
