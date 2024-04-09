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
 - `model-name` : The name of the model which will be given by the `\info` endpoint. It is solely informative and won't have any other purpose (default value is `?`)
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
 - `ssl-cert-reqs`: Uvicorn setting, Whether client certificate is required (see stdlib ssl module's) (default value is `0`)
 - `root_path`: The FastAPI root path (default value is `None`)

### Model arguments

All the usual vLLM arguments for the model and vLLM itself are usable. They all have default values (defined by vLLM) and are optional. The exhaustive list is [here (source code)](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) or [here (documentation)](https://docs.vllm.ai/en/latest/models/engine_args.html). Here are some of them:

 - `model`: The path to the model or the huggingface repository
 - `dtype`: The data type for model weights and activations.
 - `max-model-len`: The model context length. If unspecified, will be automatically derived from the model.
 - `gpu-memory-utilization`: The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. If unspecified, will use the default value of 0.9.
