FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

LABEL org.opencontainers.image.author="Agence Data Services"
LABEL org.opencontainers.image.description="REST service happy-vllm"

COPY prebuildfs /
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install python common
RUN install_packages software-properties-common

RUN add-apt-repository -d -y 'ppa:deadsnakes/ppa' \
     && install_packages python3.11 python3.11-dev python3.11-venv python3-pip gcc-10 g++-10\
     && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1\
     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=true

RUN python -m venv /opt/venv \
    && pip install --upgrade pip
ENV VIRTUAL_ENV="/opt/venv" PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Install package
COPY pyproject.toml setup.py README.md requirements.txt version.txt /app/
COPY src/happy_vllm /app/src/happy_vllm

RUN python -m pip install -r requirements.txt && python -m pip install .

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

COPY prebuildfs /
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install python common
RUN install_packages software-properties-common

RUN add-apt-repository -d -y 'ppa:deadsnakes/ppa' \
     && install_packages python3.11 python3.11-dev python3.11-venv python3-pip gcc-10 g++-10\
     && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1\
     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PIP_NO_CACHE_DIR=true

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY src/happy_vllm/launch.py /app

# Start API
EXPOSE 5000
CMD ["python", "/app/launch.py"]
