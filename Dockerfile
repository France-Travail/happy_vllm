ARG base_image=nvidia/cuda:12.1.0-devel-ubuntu22.04

# base image
FROM ${base_image}

ENV APP_NAME="happy_vllm"
ENV API_ENTRYPOINT="/happy_vllm/rs/v1"

LABEL maintainer="Agence Data Services"
LABEL description="Service REST happy_vllm"

WORKDIR /app

# Install package
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY src/ src/
COPY requirements.txt requirements.txt
COPY version.txt version.txt

RUN ln -sfn /usr/bin/python3.11 /usr/bin/python3

RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt && python3 -m pip install .

# Start API
EXPOSE 8501
CMD ["python3", "src/happy_vllm/launch.py"]
