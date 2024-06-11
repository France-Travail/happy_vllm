# Deploying with docker

The docker image is available on the [Github Container Registry](https://github.com/France-Travail/happy_vllm/pkgs/container/happy_vllm)

## Pull the image from Github Container Registry

```bash
docker pull ghcr.io/france-travail/happy_vllm:latest
```

## Launch a container

```bash
docker run -it ghcr.io/france-travail/happy_vllm:latest --model mistralai/Mistral-7B-v0.1
```
See [arguments](arguments.md) for more details the list of all arguments useful for the application and model for happy_vLLM. 

## Build docker image from source via the provided dockerfile

```bash
docker build -t france-travail/happy_vllm:latest .
```