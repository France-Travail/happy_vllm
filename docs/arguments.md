# Arguments

## CLI, environment variables and .env

Arguments for launching happy_vLLM are all defined in `utils_args.py`. They can be defined via three methods which are **by order of priority**:

 - directly from the cli (for example using the entrypoint `happy-vllm --model path_to_model`)
 - from environment variables (for example `export MODEL="path_to_model"` and then use the entrypoint `happy-vllm`)
 - from a `.env` file located where you use the `happy-vllm` entrypoint (see the `.env.example` provided for a complete description)

Note that in the cli, the arguments are lowercase and with hyphens `-` whereas in the environment variables and `.env`, they are uppercase and with underscore `_`. So for example the argument `max-model-len` in the cli corresponds to `MAX_MODEL_LEN` in the environment variables or `.env` file.

The order of priority means that the arguments entered via the cli will always overwrite the argument from environment variables and `.env` file if it is also defined there. Similarly, the environment variables will always trump the variables from a `.env` file.

## Arguments definition

### Application arguments

Here is a list of arguments useful for the application (they all have default values and are optional):

 - `host` : The name of the host (default value is `127.0.0.1`)
 - `port` : The port number (default value is `5000`)
 - `model-name` : The name of the model which will be given by the `/v1/info` endpoint or the `/v1/models`. Knowing the name of the model is important to be able to use the endpoints `/v1/completions` and `/v1/chat/completions` (default value is `?`)
 - `extra-information` : The path to a json which will be added to the `/v1/info` endpoint in the `extra_information` field 
 - `app-name`: The name of the application (default value is `happy_vllm`)
 - `api-endpoint-prefix`: The prefix added to all the API endpoints (default value is no prefix)
 - `explicit-errors`: If `False`, the message displayed when an `500 error` is encountered will be `Internal Server Error`. If `True`, the message displayed will be more explicit and give information on the underlying error. The `True` setting is not recommended in a production setting (default value is `False`).
 - `allow-credentials`: The CORS setting (default value is `False`)
 - `allowed-origins`: The CORS setting (default value is `["*"]`)
 - `allowed-methods`: The CORS setting (default value is `["*"]`)
 - `allowed-headers`: The CORS setting (default value is `["*"]`)
 - `uvicorn-log-level`: The log level of uvicorn (default value is `info`)
 - `ssl-keyfile`: Uvicorn setting, the file path to the SSL key file (default value is `None`)
 - `ssl-certfile`: Uvicorn setting, the file path to the SSL cert file (default value is `None`)
 - `ssl-ca-certs`: Uvicorn setting, the CA certificates file (default value is `None`)
 - `enable-ssl-refresh`: Refresh SSL Context when SSL certificate files change (default value is `False`)
 - `ssl-cert-reqs`: Uvicorn setting, Whether client certificate is required (see stdlib ssl module's) (default value is `0`)
 - `root_path`: The FastAPI root path (default value is `None`)
 - `lora-modules`: LoRA module configurations in the format name=path
 - `chat-template`: The file path to the chat template, or the template in single-line form for the specified model (see [the documentation of vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-template) for more details). Useful in the `/v1/chat/completions` endpoint
 - `chat-template-content-format`: The format to render message content within a chat template. `string` will render the content as a string. 'Example: "Hello World"', `openai` will render the content as a list of dictionaries, similar to OpenAI schema. 'Example: [{"type": "text", "text": "Hello world!"}]' (default value is `auto`)
 - `response-role`: The role name to return if `request.add_generation_prompt=true`. Useful in the `/v1/chat/completions` endpoint
 - `with-launch-arguments`: Whether the route `/v1/launch_arguments` gives the launch arguments or an empty json (default value is `False`)
 - `return-tokens-as-token-ids`: "When `--max-logprobs`  is specified, represents single tokens as strings of the form 'token_id:{token_id}' so that tokens that are not JSON-encodable can be identified (default value is `False`)
 - `max-log-len`: Max number of prompt characters or prompt ID numbers being printed in log (default value is unlimited)
 - `prompt-adapters`: Prompt adapter configurations in the format name=path. Multiple adapters can be specified (default value is `None`)
 - `disable-frontend-multiprocessing`: If specified, will run the OpenAI frontend server in the same process as the model serving engine (default value is `False`)
 - `enable-request-id-headers`: If specified, API server will add X-Request-Id header to responses. Caution: this hurts performance at high QPS (default value `False`)
 - `enable-auto-tool-choice`: Enable auto tool choice for supported models. Use --tool-call-parser" "to specify which parser to use" (default value is `False`)
 - `tool-call-parser`: Select the tool call parser depending on the model that you're using. This is used to parse the model-generated tool call. Required for --enable-auto-tool-choice. (default value is `None`, only `mistral` and `hermes` are allowed)
 - `tool-parser-plugin`: Special the tool parser plugin write to parse the model-generated tool into OpenAI API format, the name register in this plugin can be used in --tool-call-parser (default value is `""`)
 - `disable-fastapi-docs`: Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint (default value is `False`)
 - `enable-prompt-tokens-details`: If set to True, enable prompt_tokens_details in usage (default value is `False`)
 - `enable-server-load-tracking`: If set to True, enable tracking server_load_metrics in the app state (default value is `False`)
 - `disable-uvicorn-access-log`: Disable uvicorn access log (default value is `False`)

### Model arguments

All the usual vLLM arguments for the model and vLLM itself are usable. They all have default values (defined by vLLM) and are optional. The exhaustive list is [here (source code)](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) or [here (documentation)](https://docs.vllm.ai/en/latest/serving/engine_args.html). Here are some of them:

 - `model`: The path to the model or the huggingface repository
 - `task`: Possible choices: auto, generate, embedding, embed, classify, transcription. The task to use the model for. Each vLLM instance only supports one task, even if the same model can be used for multiple tasks. When the model only supports one task, "auto" can be used to select it; otherwise, you must specify explicitly which task to use. If unspecified, will use the default value of `auto`.
 - `dtype`: The data type for model weights and activations.
 - `max-model-len`: The model context length. If unspecified, will be automatically derived from the model.
 - `gpu-memory-utilization`: The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. If unspecified, will use the default value of 0.9.
 - `config-format`: Possible choices: auto, hf, mistral. The format of the model config to load. If unspecified, will use the default value of `ConfigFormat.AUTO`.
 - `load-format`: Possible choices: auto, pt, safetensors, npcache, dummy, tensorizer, sharded_state, gguf, bitsandbytes, mistral, runai_streamer. The format of the model weights to load. If unspecified, will use the default value of `auto`.
 - `tokenizer-mode`:Possible choices: auto, slow, mistral, custom. The tokenizer mode. If unspecified, will use the default value of `auto`.


