# Welcome to Happy_vLLM

Happy_vLLM is a REST API, production ready. It is based on the popular library [vLLM](https://github.com/vllm-project/vllm) and provide an API for it.

## Installation

You can install happy_vLLM using pip:

```bash
pip install happy_vllm
```

Or build it from source:

```bash
git clone https://github.com/France-Travail/happy_vllm.git
cd happy_vllm
pip install -e .
```

## Quickstart

Just use the entrypoint `happy-vllm` (see [arguments](arguments.md) for a list of all possible arguments)

```bash
happy-vllm --model path_to_model --host 127.0.0.1 --port 5000 --model-name my_model
```

It will launch the API and you can directly query it for example with 

```bash
curl 127.0.0.1:5000/v1/info
```

To get various information on the application or 

```bash
curl 127.0.0.1:5000/v1/completions -d '{"prompt": "Hey,", "model": "my_model"}'
```

if you want to generate your first LLM response using happy_vLLM. See [endpoints](endpoints/endpoints.md) for more details on all the endpoints provided by happy_vLLM. 

## Deploy with Docker image

A docker image is available from the [Github Container Registry](https://github.com/France-Travail/happy_vllm/pkgs/container/happy_vllm) :  

```bash
docker pull ghcr.io/france-travail/happy_vllm:latest
```
See [deploying_with_docker](deploying_with_docker.md) for more details on how to serve happy_vLLM with docker. 

## Swagger

You can reach the swagger UI at the `/docs` endpoint (so for example by default at `127.0.0.1:5000/docs`). You will be provided all the endpoints and examples on how to use them.