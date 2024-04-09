# Endpoints

happy_vLLM provides several endpoints which cover most of the use cases. Feel free to open an issue or a PR if you would like to add an endpoint. All these endpoints (except for the `/metrics` endpoint) are prefixed by, well, a prefix which by default is absent. 

## info (GET)

Provides information on the API and the model (more details [here](technical.md))

## metrics (GET)

The technical metrics obtained for prometheus (more details [here](technical.md))

## liveness (GET)

The liveness endpoint (more details [here](technical.md))

## readiness (GET)

The readiness endpoint (more details [here](technical.md))

## generate and generate_stream (POST)

The core of the reactor. These two routes take a prompt and completes it (more details [here](generate.md))

## tokenizer (POST)

Used to tokenizer a text (more details [here](tokenizer.md))

## decode (POST)

Used to decode a list of token ids (more details [here](tokenizer.md))

## metadata_text (POST)

Used to know which part of a prompt will be truncated (more details [here](data_manipulation.md))

## split_text (POST)

Splits a text on some separators, for example to prepare for some RAG (more details [here](data_manipulation.md))