# syntax=docker/dockerfile:1
# Official PyTorch image, Linux amd64, Python 3.11, CUDA 12.8.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385 AS runtime-libs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1
WORKDIR /service
COPY requirements.lock /service/requirements.lock
RUN python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
RUN python -c "import torch, torchvision, transformers; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.8'; assert torchvision.__version__.split('+')[0] == '0.23.0'; assert transformers.__version__ == '4.56.2'; assert 'sm_120' in torch._C._cuda_getArchFlags().split(); print('CUDA package and sm_120 build checks passed; GPU inference is not tested')"

FROM runtime-libs AS model-download
COPY model_pins.py download_models.py /service/
RUN python download_models.py --destination /opt/models

FROM runtime-libs AS service
COPY --from=model-download /opt/models /opt/models
COPY app.py backend.py contracts.py model_pins.py download_models.py /service/
COPY NOTICE.md /service/NOTICE.md
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    HF_MODULES_CACHE=/tmp/huggingface/modules \
    MODEL_ROOT=/opt/models
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --start-period=5m --interval=30s --timeout=3s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "16", "--timeout-keep-alive", "5"]
