# Endpoints

happy_vLLM provides several endpoints which cover most of the use cases. Feel free to open an issue or a PR if you would like to add an endpoint. All these endpoints (except for the `/metrics` endpoint) are prefixed by, well, a prefix which by default is absent. 

## Technical endpoints

### /v1/info (GET)

Provides information on the API and the model (more details [here](technical.md))

### /metrics (GET)

The technical metrics obtained for prometheus (more details [here](technical.md))

### /liveness (GET)

The liveness endpoint (more details [here](technical.md))

### /readiness (GET)

The readiness endpoint (more details [here](technical.md))

### /v1/models (GET)

The Open AI compatible endpoint used, for example, to get the name of the model. Mimicks the vLLM implementation (more details [here](technical.md))

### /v1/launch_arguments (GET)

Gives all the arguments used when launching the application. `--with-launch-arguments` must be activated (more details [here](technical.md))

## Generating endpoints

### /v1/completions and /v1/chat/completions (POST)

These two endpoints mimick the ones of vLLM. They follow the Open AI contract and you can find more details in [the vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

### /v1/abort_request (POST)

Aborts a running request 

### DEPRECATED /v1/generate and /v1/generate_stream (POST)

DEPRECATED : These two routes take a prompt and completes it (more details [here](generate.md))

## Tokenizer endpoints

### /v1/tokenizer (POST) :warning: **Deprecated**

Used to tokenizer a text (more details [here](tokenizer.md))

### /v2/tokenizer (POST)

Used to tokenizer a text (more details [here](tokenizer.md))

### /v1/decode (POST) :warning: **Deprecated**

Used to decode a list of token ids (more details [here](tokenizer.md))

### /v2/decode (POST)

Used to decode a list of token ids (more details [here](tokenizer.md))

## Data manipulation endpoints

### /v1/metadata_text (POST)

Used to know which part of a prompt will be truncated (more details [here](data_manipulation.md))

### /v1/split_text (POST)

Splits a text on some separators, for example to prepare for some RAG (more details [here](data_manipulation.md))