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

import sys
import ssl
import json
import torch

from argparse import Namespace, BooleanOptionalAction
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Tuple, Union, List, Mapping, Dict, Any

from vllm.utils import FlexibleArgumentParser
from vllm.executor.executor_base import ExecutorBase
from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.entrypoints.openai.cli_args import LoRAParserAction, PromptAdapterParserAction
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import BaseTokenizerGroup


DEFAULT_MODEL_NAME = '?'
DEFAULT_APP_NAME = "happy_vllm"
DEFAULT_API_ENDPOINT_PREFIX = ""
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
DEFAULT_LORA_MODULES = None
DEFAULT_CHAT_TEMPLATE = None
DEFAULT_RESPONSE_ROLE = "assistant"
DEFAULT_WITH_LAUNCH_ARGUMENTS = False
DEFAULT_MAX_LOG_LEN = None
DEFAULT_PROMPT_ADAPTERS = None
DEFAULT_RETURN_TOKENS_AS_TOKEN_IDS = False
DEFAULT_DISABLE_FRONTEND_MULTIPROCESSING = False
DEFAULT_ENABLE_AUTO_TOOL_CHOICE = False
DEFAULT_TOOL_CALL_PARSER = None


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
    lora_modules: Optional[str] = DEFAULT_LORA_MODULES
    chat_template : Optional[str] = DEFAULT_CHAT_TEMPLATE
    response_role: str = DEFAULT_RESPONSE_ROLE
    with_launch_arguments: bool = DEFAULT_WITH_LAUNCH_ARGUMENTS
    max_log_len: Optional[int] = DEFAULT_MAX_LOG_LEN
    prompt_adapters: Optional[str] = DEFAULT_PROMPT_ADAPTERS
    return_tokens_as_token_ids: bool = DEFAULT_RETURN_TOKENS_AS_TOKEN_IDS
    disable_frontend_multiprocessing: bool = DEFAULT_DISABLE_FRONTEND_MULTIPROCESSING
    enable_auto_tool_choice: bool = DEFAULT_ENABLE_AUTO_TOOL_CHOICE
    tool_call_parser: Optional[str] = DEFAULT_TOOL_CALL_PARSER


    model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))


def get_model_settings(parser: FlexibleArgumentParser) -> BaseSettings:
    """Gets the model settings. It corresponds to the variables added via AsyncEngineArgs.add_cli_args plus model-name.
    First we use the parser to get the default values of vLLM for these variables. We instantiate a BaseSettings model
    with these values as default. They are possibly overwritten by environnement variables or those of a .env

    Args:
        parser (FlexibleArgumentParser) : The parser containing all the model variables with thei default values from vLLM 
    """

    default_args = parser.parse_args([])

    # Define ModelSettings with the default values of the args (which will be replaced)
    # by the environnement variables or those of the .env file
    class ModelSettings(BaseSettings):
        model: str = default_args.model
        model_name: str = default_args.model_name
        served_model_name: Optional[Union[str, List[str]]] = None
        tokenizer: Optional[str] = default_args.tokenizer
        skip_tokenizer_init: bool = False
        tokenizer_mode: str = default_args.tokenizer_mode
        trust_remote_code: bool = False
        download_dir: Optional[str] = default_args.download_dir
        load_format: str = default_args.load_format
        config_format: str = default_args.config_format
        dtype: str = default_args.dtype
        kv_cache_dtype: str = default_args.kv_cache_dtype
        quantization_param_path: Optional[str] = default_args.quantization_param_path
        seed: int = default_args.seed
        max_model_len: Optional[int] = default_args.max_model_len
        worker_use_ray: bool = False
        distributed_executor_backend: Optional[Union[str, ExecutorBase]] = default_args.distributed_executor_backend
        pipeline_parallel_size: int = default_args.pipeline_parallel_size
        tensor_parallel_size: int = default_args.tensor_parallel_size
        max_parallel_loading_workers: Optional[int] = default_args.max_parallel_loading_workers
        block_size: int = default_args.block_size
        enable_prefix_caching: bool = False
        disable_sliding_window: bool = False
        swap_space: float = default_args.swap_space # GiB
        cpu_offload_gb: float = default_args.cpu_offload_gb  # GiB
        gpu_memory_utilization: float = default_args.gpu_memory_utilization
        max_num_batched_tokens: Optional[int] = default_args.max_num_batched_tokens
        max_num_seqs: int = default_args.max_num_seqs
        disable_log_stats: bool = False
        revision: Optional[str] = default_args.revision
        code_revision: Optional[str] = default_args.code_revision
        rope_scaling: Optional[dict] = default_args.rope_scaling
        rope_theta: Optional[float] = None
        tokenizer_revision: Optional[str] = default_args.tokenizer_revision
        quantization: Optional[str] = default_args.quantization
        enforce_eager: Optional[bool] = default_args.enforce_eager
        max_context_len_to_capture: Optional[int] = default_args.max_context_len_to_capture
        max_seq_len_to_capture: int = default_args.max_seq_len_to_capture
        disable_custom_all_reduce: bool = False
        enable_lora: bool = False
        max_loras: int = default_args.max_loras
        max_lora_rank: int = default_args.max_lora_rank
        enable_prompt_adapter: bool = False
        max_prompt_adapters: int = default_args.max_prompt_adapters
        max_prompt_adapter_token: int = default_args.max_prompt_adapter_token
        fully_sharded_loras: bool = False
        lora_extra_vocab_size: int = default_args.lora_extra_vocab_size
        long_lora_scaling_factors: Optional[Tuple[float]] = default_args.long_lora_scaling_factors
        lora_dtype: Optional[Union[str, torch.dtype]] = default_args.lora_dtype
        max_cpu_loras: Optional[int] = default_args.max_cpu_loras
        device: str = default_args.device
        num_scheduler_steps: int = default_args.num_scheduler_steps
        ray_workers_use_nsight: bool = False
        num_gpu_blocks_override: Optional[int] = default_args.num_gpu_blocks_override
        num_lookahead_slots: int = default_args.num_lookahead_slots
        model_loader_extra_config: Optional[dict] = default_args.model_loader_extra_config
        ignore_patterns:  Optional[Union[str, List[str]]] = default_args.ignore_patterns
        preemption_mode: Optional[str] = default_args.preemption_mode
        disable_log_requests: bool = False
        engine_use_ray: bool = False
        use_v2_block_manager: bool = False
        max_logprobs: int = default_args.max_logprobs
        tokenizer_pool_size: int = default_args.tokenizer_pool_size
        tokenizer_pool_type: Union[str, BaseTokenizerGroup] = default_args.tokenizer_pool_type
        tokenizer_pool_extra_config: Optional[str] = default_args.tokenizer_pool_extra_config
        limit_mm_per_prompt: Optional[Mapping[str, int]] = default_args.limit_mm_per_prompt
        scheduler_delay_factor: float = default_args.scheduler_delay_factor
        enable_chunked_prefill: Optional[bool] = default_args.enable_chunked_prefill
        guided_decoding_backend: str = default_args.guided_decoding_backend
        # Speculative decoding configuration.
        speculative_model: Optional[str] = default_args.speculative_model
        speculative_model_quantization: Optional[str] = default_args.speculative_model_quantization
        speculative_draft_tensor_parallel_size: Optional[int] = default_args.speculative_draft_tensor_parallel_size
        num_speculative_tokens: Optional[int] = default_args.num_speculative_tokens
        speculative_max_model_len: Optional[int] = default_args.speculative_max_model_len
        speculative_disable_by_batch_size: Optional[int] = default_args.speculative_disable_by_batch_size
        ngram_prompt_lookup_max: Optional[int] = default_args.ngram_prompt_lookup_max
        ngram_prompt_lookup_min: Optional[int] = default_args.ngram_prompt_lookup_min
        spec_decoding_acceptance_method: str = default_args.spec_decoding_acceptance_method
        typical_acceptance_sampler_posterior_threshold: Optional[float] = default_args.typical_acceptance_sampler_posterior_threshold
        typical_acceptance_sampler_posterior_alpha: Optional[float] = default_args.typical_acceptance_sampler_posterior_alpha
        qlora_adapter_name_or_path: Optional[str] = default_args.qlora_adapter_name_or_path
        disable_logprobs_during_spec_decoding: Optional[bool] = default_args.disable_logprobs_during_spec_decoding
        disable_async_output_proc: bool = False
        override_neuron_config: Optional[Dict[str, Any]] = default_args.override_neuron_config

        otlp_traces_endpoint: Optional[str] = default_args.otlp_traces_endpoint
        collect_detailed_traces: Optional[str] = default_args.collect_detailed_traces

        model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))

    model_settings = ModelSettings()

    return model_settings


def get_parser() -> FlexibleArgumentParser:
    """Gets the parser. The default values of all application variables (see ApplicationSettings) are properly
    set to the BaseSetting value defined via pydantic. The default values of all model variables (ie those added
    via AsyncEngineArgs.add_cli_args plus model-name) are not properly set via pydantic at this point.
    
    Returns:
        FlexibleArgumentParser : The argparse parser
    """
    parser = FlexibleArgumentParser(description="REST API server for vLLM, production ready")

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
                        help="allow credentials for CORS")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=application_settings.allowed_origins,
                        help="CORS allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=application_settings.allowed_methods,
                        help="CORS allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=application_settings.allowed_headers,
                        help="CORS allowed headers")
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
    parser.add_argument("--lora-modules",
                        type=str,
                        default=application_settings.lora_modules,
                        nargs='+',
                        action=LoRAParserAction,
                        help="LoRA module configurations in the format name=path. "
                        "Multiple modules can be specified.")
    parser.add_argument("--chat-template",
                        type=str,
                        default=application_settings.chat_template,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model")
    parser.add_argument("--response-role",
                        type=str,
                        default=application_settings.response_role,
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument("--with-launch-arguments",
                        type=bool,
                        default=application_settings.with_launch_arguments,
                        help="Whether the route launch_arguments should display the launch arguments")
    parser.add_argument('--max-log-len',
                        type=int,
                        default=application_settings.max_log_len,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        '\n\nDefault: Unlimited')
    parser.add_argument(
                "--prompt-adapters",
                type=nullable_str,
                default=application_settings.prompt_adapters,
                nargs='+',
                action=PromptAdapterParserAction,
                help="Prompt adapter configurations in the format name=path. "
                "Multiple adapters can be specified.")
    parser.add_argument(
        "--return-tokens-as-token-ids",
        default=application_settings.return_tokens_as_token_ids,
        action=BooleanOptionalAction,
        help="When --max-logprobs is specified, represents single tokens as "
        "strings of the form 'token_id:{token_id}' so that tokens that "
        "are not JSON-encodable can be identified.")
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        default=application_settings.disable_frontend_multiprocessing,
        action=BooleanOptionalAction,
        help="If specified, will run the OpenAI frontend server in the same "
        "process as the model serving engine.")
    parser.add_argument(
        "--enable-auto-tool-choice",
        default=application_settings.enable_auto_tool_choice,
        action=BooleanOptionalAction,
        help=
        "Enable auto tool choice for supported models. Use --tool-call-parser"
        "to specify which parser to use")
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        choices=["mistral", "hermes"],
        default=application_settings.tool_call_parser,
        help=
        "Select the tool call parser depending on the model that you're using."
        " This is used to parse the model-generated tool call into OpenAI API "
        "format. Required for --enable-auto-tool-choice.")

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

    # Explicitly check for help flag for the providing help message to the entrypoint
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        parser.print_help()
        sys.exit()
    return args
