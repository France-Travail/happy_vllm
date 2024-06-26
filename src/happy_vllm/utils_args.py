#!/usr/bin/env python3
# Copyright (C) <2018-2024>  <Agence Data Services, DSI France Travail>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import ssl
import json

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from argparse import Namespace, ArgumentParser, BooleanOptionalAction

from vllm.engine.arg_utils import AsyncEngineArgs


DEFAULT_MODEL_NAME = '?'
DEFAULT_APP_NAME = "happy_vllm"
DEFAULT_API_ENDPOINT_PREFIX = "/happy_vllm/rs/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
DEFAULT_EXPLICIT_ERRORS = False
DEFAULT_ALLOW_CREDENTIALS = False
DEFAULT_ALLOWED_ORIGINS = ["*"]
DEFAULT_ALLOWED_METHODS = ["*"]
DEFAULT_ALLOWED_HEADERS = ["*"]
DEFAULT_UVICORN_LOG_LEVEL = 'info'
CHOICES_UVICORN_LOG_LEVEL = ['debug', 'info', 'warning', 'error', 'critical', 'trace']
DEFAULT_SSL_KEYFILE = None
DEFAULT_SSL_CERTFILE = None
DEFAULT_SSL_CA_CERTS = None
DEFAULT_SSL_CERT_REQS = int(ssl.CERT_NONE)
DEFAULT_ROOT_PATH = None


class ApplicationSettings(BaseSettings):
    """Application settings

    This class is used for settings management purpose, have a look at the pydantic
    documentation for more details : https://pydantic-docs.helpmanual.io/usage/settings/

    By default, it looks for environment variables (case insensitive) to set the settings
    if a variable is not found, it looks for a file name .env in your working directory
    where you can declare the values of the variables and finally it sets the values
    to the default ones one can define above
    """
    host : str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    explicit_errors: bool = DEFAULT_EXPLICIT_ERRORS
    allow_credentials: bool = DEFAULT_ALLOW_CREDENTIALS
    allowed_origins: list = DEFAULT_ALLOWED_ORIGINS
    allowed_methods: list = DEFAULT_ALLOWED_METHODS
    allowed_headers: list = DEFAULT_ALLOWED_HEADERS
    uvicorn_log_level: str = DEFAULT_UVICORN_LOG_LEVEL
    ssl_keyfile: Optional[str] = DEFAULT_SSL_KEYFILE
    ssl_certfile: Optional[str] = DEFAULT_SSL_CERTFILE
    ssl_ca_certs: Optional[str] = DEFAULT_SSL_CA_CERTS
    ssl_cert_reqs: int = DEFAULT_SSL_CERT_REQS
    root_path: Optional[str] = DEFAULT_ROOT_PATH
    app_name: str = DEFAULT_APP_NAME
    api_endpoint_prefix: str = DEFAULT_API_ENDPOINT_PREFIX

    model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))


def get_model_settings(parser: ArgumentParser) -> BaseSettings:
    """Gets the model settings. It corresponds to the variables added via AsyncEngineArgs.add_cli_args plus model-name.
    First we use the parser to get the default values of vLLM for these variables. We instantiate a BaseSettings model
    with these values as default. They are possibly overwritten by environnement variables or those of a .env

    Args:
        parser (ArgumentParser) : The parser containing all the model variables with thei default values from vLLM 
    """

    default_args = parser.parse_args([])

    # Define ModelSettings with the default values of the args (which will be replaced)
    # by the environnement variables or those of the .env file
    class ModelSettings(BaseSettings):
        model: str = default_args.model
        model_name: str = default_args.model_name
        tokenizer: Optional[str] = default_args.tokenizer
        tokenizer_mode: str = default_args.tokenizer_mode
        trust_remote_code: bool = False
        download_dir: Optional[str] = default_args.download_dir
        load_format: str = default_args.load_format
        dtype: str = default_args.dtype
        kv_cache_dtype: str = default_args.kv_cache_dtype
        seed: int = default_args.seed
        max_model_len: Optional[int] = default_args.max_model_len
        worker_use_ray: bool = False
        pipeline_parallel_size: int = default_args.pipeline_parallel_size
        tensor_parallel_size: int = default_args.tensor_parallel_size
        max_parallel_loading_workers: Optional[int] = default_args.max_parallel_loading_workers
        block_size: int = default_args.block_size
        enable_prefix_caching: bool = False
        swap_space: int = default_args.swap_space
        gpu_memory_utilization: float = default_args.gpu_memory_utilization
        max_num_batched_tokens: Optional[int] = default_args.max_num_batched_tokens
        max_num_seqs: int = default_args.max_num_seqs
        disable_log_stats: bool = False
        revision: Optional[str] = default_args.revision
        code_revision: Optional[str] = default_args.code_revision
        tokenizer_revision: Optional[str] = default_args.tokenizer_revision
        quantization: Optional[str] = default_args.quantization
        enforce_eager: bool = False
        max_context_len_to_capture: int = default_args.max_context_len_to_capture
        disable_custom_all_reduce: bool = False
        enable_lora: bool = False
        max_loras: int = default_args.max_loras
        max_lora_rank: int = default_args.max_lora_rank
        lora_extra_vocab_size: int = default_args.lora_extra_vocab_size
        lora_dtype: str = default_args.lora_dtype
        max_cpu_loras: Optional[int] = default_args.max_cpu_loras
        device: str = default_args.device
        ray_workers_use_nsight: bool = False
        max_log_len: Optional[int] = default_args.max_log_len
        disable_log_requests: bool = False
        engine_use_ray: bool = False

        model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))

    model_settings = ModelSettings()

    return model_settings


def get_parser() -> ArgumentParser:
    """Gets the parser. The default values of all application variables (see ApplicationSettings) are properly
    set to the BaseSetting value defined via pydantic. The default values of all model variables (ie those added
    via AsyncEngineArgs.add_cli_args plus model-name) are not properly set via pydantic at this point.
    
    Returns:
        ArgumentParser : The argparse parser
    """
    parser = ArgumentParser(description="REST API server for vLLM, production ready")

    application_settings = ApplicationSettings(_env_parse_none_str='None') # type: ignore

    parser.add_argument("--host",
                        type=str,
                        default=application_settings.host,
                        help="host name")
    parser.add_argument("--port",
                        type=int,
                        default=application_settings.port,
                        help="port number")
    parser.add_argument("--model-name",
                        type=str,
                        default=DEFAULT_MODEL_NAME,
                        help="The name of the model given by the /info endpoint of the API")
    parser.add_argument("--app-name",
                        type=str,
                        default=application_settings.app_name,
                        help="The name of the application")
    parser.add_argument("--api-endpoint-prefix",
                        type=str,
                        default=application_settings.api_endpoint_prefix,
                        help="The prefix for the API endpoints")
    parser.add_argument("--explicit-errors",
                        default=application_settings.explicit_errors,
                        action=BooleanOptionalAction,
                        help="If True, the underlying python errors are sent back via the API")
    parser.add_argument('--allow-credentials',
                        default=application_settings.allow_credentials,
                        action=BooleanOptionalAction,
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=application_settings.allowed_origins,
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=application_settings.allowed_methods,
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=application_settings.allowed_headers,
                        help="allowed headers")
    parser.add_argument("--uvicorn-log-level",
                        type=str,
                        default=application_settings.uvicorn_log_level,
                        choices=CHOICES_UVICORN_LOG_LEVEL,
                        help="log level for uvicorn")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=application_settings.ssl_keyfile,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=application_settings.ssl_certfile,
                        help="The file path to the SSL cert file")
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=application_settings.ssl_ca_certs,
                        help="The CA certificates file")
    parser.add_argument("--ssl-cert-reqs",
                        type=int,
                        default=application_settings.ssl_cert_reqs,
                        help="Whether client certificate is required (see stdlib ssl module's)")
    parser.add_argument("--root-path",
                        type=str,
                        default=application_settings.root_path,
                        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)

    return parser


def parse_args() -> Namespace:
    """Parses the args. We want this priority : args from cli > environnement variables > .env variables.
    In order to do this, we get from pydantic BaseSetting the value from environnement variables or .env variables
    with the proper priority. Then we set the default value of the cli parser to those value so that if the cli args
    are used, they overwrite the default values and otherwise, the BaseSetting value is taken.

    Returns:
        NameSpace
    """
    # Gets the parser
    # The default value of the application variables are properly set
    # Those of the model variables are not
    parser = get_parser()
    # Gets the default values of the model variables via pydantic
    model_settings = get_model_settings(parser)
    # Sets the default values of the model variables in the parser
    parser.set_defaults(**model_settings.model_dump())
    # Gets the args
    args = parser.parse_args()
    return args
