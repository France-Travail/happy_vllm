![happy_vllm logo](https://raw.githubusercontent.com/France-Travail/happy_vllm/main/docs/source/assets/logo/logo_happy_vllm.svg)

[![pypi badge](https://img.shields.io/pypi/v/happy_vllm.svg)](https://pypi.python.org/pypi/happy_vllm)
[![Generic badge](https://img.shields.io/badge/python-3.10|3.11-blue.svg)](https://shields.io/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

![Build & Tests](https://github.com/France-Travail/happy_vllm/actions/workflows/build_and_tests.yaml/badge.svg)
![Wheel setup](https://github.com/France-Travail/happy_vllm/actions/workflows/wheel.yaml/badge.svg)
![docs](https://github.com/France-Travail/happy_vllm/actions/workflows/docs.yaml/badge.svg)


**📚 Documentation :** [https://france-travail.github.io/happy_vllm/](https://france-travail.github.io/happy_vllm/) <!-- omit in toc -->

---

happy_vLLM is a REST API for [vLLM](https://github.com/vllm-project/vllm) which was developed with production in mind. It adds some [functionalities](https://france-travail.github.io/happy_vllm/pros/) to vLLM.

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

Just use the entrypoint `happy-vllm` (see [arguments](https://france-travail.github.io/happy_vllm/arguments/) for a list of all possible arguments)

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

if you want to generate your first LLM response using happy_vLLM. See [endpoints](https://france-travail.github.io/happy_vllm/endpoints/endpoints) for more details on all the endpoints provided by happy_vLLM. 

## Deploy with Docker image

A docker image is available from the [Github Container Registry](https://github.com/France-Travail/happy_vllm/pkgs/container/happy_vllm) :  

```bash
docker pull ghcr.io/france-travail/happy_vllm:latest
```
See [deploying_with_docker](https://france-travail.github.io/happy_vllm/deploying_with_docker) for more details on how to serve happy_vLLM with docker. 

## Swagger

You can reach the swagger UI at the `/docs` endpoint (so for example by default at `127.0.0.1:5000/docs`). You will be displayed all the endpoints and examples on how to use them.