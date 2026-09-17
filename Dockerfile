FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS reg2026_algorithm_amd64
# PyTorch + CUDA base so GPU is available at inference time.
# cu12.8 build (same torch 2.9.1) ships sm_70..sm_120 kernels: runs on T4/A100/H100
# AND newer sm_120 GPUs. cu12.6 only shipped up to sm_90, so sm_120 hit
# "no kernel image available for execution on the device".

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
# Ensure imports like "from core import ..." work from src/ submodules
ENV PYTHONPATH=/opt/app

# git is needed at build time to pip-install TRIDENT from its GitHub repo.
# (openslide is provided by the openslide-bin pip wheel - no apt openslide needed.)
# git: pip-install TRIDENT from GitHub. libvips-tools: the `vips` CLI used by interf1 to
# convert single-level platform WSIs into tiled pyramids (fast seg/patch/feat reads).
RUN apt-get update && apt-get install -y --no-install-recommends git libvips-tools && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# TRIDENT pulls opencv-python (full), which needs GUI libs (libGL/libxcb) absent
# from the slim runtime base. Swap to the headless build (no GUI deps, same cv2 API).
RUN python -m pip uninstall -y opencv-python opencv-contrib-python || true && \
    python -m pip install --user --no-cache-dir --no-color opencv-python-headless

COPY --chown=user:user core.py      /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user src/         /opt/app/src/

ENTRYPOINT ["python", "inference.py"]
