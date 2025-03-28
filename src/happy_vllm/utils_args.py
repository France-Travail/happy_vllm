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
from typing import Optional, Tuple, Type, Union, List, Mapping, Dict, Any, Literal, get_args

from vllm.utils import FlexibleArgumentParser
from vllm.executor.executor_base import ExecutorBase
from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption)
from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager
from vllm.entrypoints.openai.cli_args import LoRAParserAction, PromptAdapterParserAction
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import BaseTokenizerGroup
from vllm.config import (ConfigFormat, TaskOption, HfOverrides, PoolerConfig, CompilationConfig, KVTransferConfig)


DEFAULT_MODEL_NAME = '?'
DEFAULT_EXTRA_INFORMATION = None
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
DEFAULT_ENABLE_SSL_REFRESH = False
DEFAULT_ROOT_PATH = None
DEFAULT_LORA_MODULES = None
DEFAULT_CHAT_TEMPLATE = None
DEFAULT_CHAT_TEMPLATE_CONTENT_FORMAT = "auto"
DEFAULT_RESPONSE_ROLE = "assistant"
DEFAULT_WITH_LAUNCH_ARGUMENTS = False
DEFAULT_MAX_LOG_LEN = None
DEFAULT_PROMPT_ADAPTERS = None
DEFAULT_RETURN_TOKENS_AS_TOKEN_IDS = False
DEFAULT_DISABLE_FRONTEND_MULTIPROCESSING = False
DEFAULT_ENABLE_REQUEST_ID_HEADERS = False
DEFAULT_ENABLE_AUTO_TOOL_CHOICE = False
DEFAULT_TOOL_CALL_PARSER = None
DEFAULT_TOOL_PARSER_PLUGIN = ""
DEFAULT_DISABLE_FASTAPI_DOCS = False
DEFAULT_ENABLE_PROMPT_TOKENS_DETAILS = False
DEFAULT_ENABLE_SERVER_LOAD_TRACKING = False
DEFAULT_DISABLE_UVICORN_ACCESS_LOG = False

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
    enable_ssl_refresh: bool = DEFAULT_ENABLE_SSL_REFRESH
    root_path: Optional[str] = DEFAULT_ROOT_PATH
    app_name: str = DEFAULT_APP_NAME
    api_endpoint_prefix: str = DEFAULT_API_ENDPOINT_PREFIX
    lora_modules: Optional[str] = DEFAULT_LORA_MODULES
    chat_template : Optional[str] = DEFAULT_CHAT_TEMPLATE
    chat_template_content_format: str = DEFAULT_CHAT_TEMPLATE_CONTENT_FORMAT
    response_role: str = DEFAULT_RESPONSE_ROLE
    with_launch_arguments: bool = DEFAULT_WITH_LAUNCH_ARGUMENTS
    max_log_len: Optional[int] = DEFAULT_MAX_LOG_LEN
    prompt_adapters: Optional[str] = DEFAULT_PROMPT_ADAPTERS
    return_tokens_as_token_ids: bool = DEFAULT_RETURN_TOKENS_AS_TOKEN_IDS
    disable_frontend_multiprocessing: bool = DEFAULT_DISABLE_FRONTEND_MULTIPROCESSING
    enable_request_id_headers: bool = DEFAULT_ENABLE_REQUEST_ID_HEADERS
    enable_auto_tool_choice: bool = DEFAULT_ENABLE_AUTO_TOOL_CHOICE
    tool_call_parser: Optional[str] = DEFAULT_TOOL_CALL_PARSER
    tool_parser_plugin: Optional[str] = DEFAULT_TOOL_PARSER_PLUGIN
    disable_fastapi_docs : Optional[bool] = DEFAULT_DISABLE_FASTAPI_DOCS
    enable_prompt_tokens_details : Optional[bool] = DEFAULT_ENABLE_PROMPT_TOKENS_DETAILS
    enable_server_load_tracking: Optional[bool]= DEFAULT_ENABLE_SERVER_LOAD_TRACKING
    disable_uvicorn_access_log: bool = DEFAULT_DISABLE_UVICORN_ACCESS_LOG


    model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))


def get_model_settings(parser: FlexibleArgumentParser) -> BaseSettings:
    """Gets the model settings. It corresponds to the variables added via AsyncEngineArgs.add_cli_args plus model-name and extra-information.
    First we use the parser to get the default values of vLLM for these variables. We instantiate a BaseSettings model
    with these values as default. They are possibly overwritten by environnement variables or those of a .env

    Args:
        parser (FlexibleArgumentParser) : The parser containing all the model variables with their default values from vLLM 
    """

    default_args = parser.parse_args([])
    # Define ModelSettings with the default values of the args (which will be replaced)
    # by the environnement variables or those of the .env file
    class ModelSettings(BaseSettings):
        model: str = default_args.model
        model_name: str = default_args.model_name
        extra_information: Optional[str] = default_args.extra_information
        served_model_name: Optional[Union[str, List[str]]] = None
        tokenizer: Optional[str] = default_args.tokenizer
        hf_config_path: Optional[str] = default_args.hf_config_path
        task: TaskOption = default_args.task
        skip_tokenizer_init: bool = False
        tokenizer_mode: str = default_args.tokenizer_mode
        trust_remote_code: bool = False

        allowed_local_media_path: Optional[str] = default_args.allowed_local_media_path
        download_dir: Optional[str] = default_args.download_dir
        load_format: str = default_args.load_format
        config_format: ConfigFormat = default_args.config_format
        dtype: str = default_args.dtype
        kv_cache_dtype: str = default_args.kv_cache_dtype
        seed: Optional[int] = default_args.seed
        max_model_len: Optional[int] = default_args.max_model_len
        distributed_executor_backend: Optional[Union[str, ExecutorBase]] = default_args.distributed_executor_backend
        pipeline_parallel_size: int = default_args.pipeline_parallel_size
        tensor_parallel_size: int = default_args.tensor_parallel_size
        enable_expert_parallel: bool = False
        max_parallel_loading_workers: Optional[int] = default_args.max_parallel_loading_workers
        block_size: Optional[int] = default_args.block_size
        enable_prefix_caching: Optional[bool] = default_args.enable_prefix_caching
        disable_sliding_window: bool = False
        disable_cascade_attn: bool = False
        swap_space: float = default_args.swap_space # GiB
        cpu_offload_gb: float = default_args.cpu_offload_gb  # GiB
        gpu_memory_utilization: float = default_args.gpu_memory_utilization
        max_num_batched_tokens: Optional[int] = default_args.max_num_batched_tokens
        max_num_partial_prefills: Optional[int] = default_args.max_num_partial_prefills
        max_long_partial_prefills: Optional[int] = default_args.max_long_partial_prefills
        long_prefill_token_threshold: Optional[int] = default_args.long_prefill_token_threshold
        max_num_seqs: Optional[int] = default_args.max_num_seqs
        disable_log_stats: bool = False
        revision: Optional[str] = default_args.revision
        code_revision: Optional[str] = default_args.code_revision
        rope_scaling: Optional[Dict[str, Any]] = default_args.rope_scaling
        rope_theta: Optional[float] = None
        hf_overrides: Optional[HfOverrides] = default_args.hf_overrides
        tokenizer_revision: Optional[str] = default_args.tokenizer_revision
        quantization: Optional[str] = default_args.quantization
        enforce_eager: Optional[bool] = default_args.enforce_eager
        max_seq_len_to_capture: int = default_args.max_seq_len_to_capture
        disable_custom_all_reduce: bool = False
        enable_lora: bool = False
        enable_lora_bias: bool = False
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
        multi_step_stream_outputs: bool = default_args.multi_step_stream_outputs
        ray_workers_use_nsight: bool = False
        num_gpu_blocks_override: Optional[int] = default_args.num_gpu_blocks_override
        num_lookahead_slots: int = default_args.num_lookahead_slots
        model_loader_extra_config: Optional[dict] = default_args.model_loader_extra_config
        ignore_patterns:  Optional[Union[str, List[str]]] = default_args.ignore_patterns
        preemption_mode: Optional[str] = default_args.preemption_mode
        disable_log_requests: bool = False
        engine_use_ray: bool = False
        use_v2_block_manager: bool = default_args.use_v2_block_manager
        max_logprobs: int = default_args.max_logprobs
        tokenizer_pool_size: int = default_args.tokenizer_pool_size
        tokenizer_pool_type: Union[str, BaseTokenizerGroup] = default_args.tokenizer_pool_type
        tokenizer_pool_extra_config: Optional[Dict[str, Any]] = default_args.tokenizer_pool_extra_config
        limit_mm_per_prompt: Optional[Mapping[str, int]] = default_args.limit_mm_per_prompt
        mm_processor_kwargs: Optional[Dict[str, Any]] = default_args.mm_processor_kwargs
        disable_mm_preprocessor_cache: bool = False
        scheduling_policy: Literal["fcfs", "priority"] = default_args.scheduling_policy
        scheduler_cls: Union[str, Type[object]] = default_args.scheduler_cls
        scheduler_delay_factor: float = default_args.scheduler_delay_factor
        enable_chunked_prefill: Optional[bool] = default_args.enable_chunked_prefill
        guided_decoding_backend: str = default_args.guided_decoding_backend
        logits_processor_pattern: Optional[str] = default_args.logits_processor_pattern
        speculative_config: Optional[Union[str, Dict[str, Any]]] = default_args.speculative_config
        speculative_model: Optional[str] = default_args.speculative_model
        speculative_model_quantization: Optional[str] = default_args.speculative_model_quantization
        speculative_draft_tensor_parallel_size: Optional[int] = default_args.speculative_draft_tensor_parallel_size
        num_speculative_tokens: Optional[int] = default_args.num_speculative_tokens
        speculative_disable_mqa_scorer: Optional[bool] = default_args.speculative_disable_mqa_scorer
        speculative_max_model_len: Optional[int] = default_args.speculative_max_model_len
        speculative_disable_by_batch_size: Optional[int] = default_args.speculative_disable_by_batch_size
        ngram_prompt_lookup_max: Optional[int] = default_args.ngram_prompt_lookup_max
        ngram_prompt_lookup_min: Optional[int] = default_args.ngram_prompt_lookup_min
        spec_decoding_acceptance_method: str = default_args.spec_decoding_acceptance_method
        typical_acceptance_sampler_posterior_threshold: Optional[float] = default_args.typical_acceptance_sampler_posterior_threshold
        typical_acceptance_sampler_posterior_alpha: Optional[float] = default_args.typical_acceptance_sampler_posterior_alpha
        disable_logprobs_during_spec_decoding: Optional[bool] = default_args.disable_logprobs_during_spec_decoding
        qlora_adapter_name_or_path: Optional[str] = default_args.qlora_adapter_name_or_path
        show_hidden_metrics_for_version: Optional[str] = default_args.show_hidden_metrics_for_version
        disable_async_output_proc: bool = False
        override_neuron_config: Optional[Dict[str, Any]] = default_args.override_neuron_config
        override_pooler_config: Optional[PoolerConfig] = default_args.override_pooler_config
        compilation_config: Optional[CompilationConfig] = default_args.compilation_config
        worker_cls: str = default_args.worker_cls
        worker_extension_cls: str = default_args.worker_extension_cls
        kv_transfer_config: Optional[KVTransferConfig] = default_args.kv_transfer_config
        generation_config: Optional[str] = default_args.generation_config
        override_generation_config: Optional[Dict[str, Any]] = default_args.override_generation_config
        enable_sleep_mode: bool = False
        model_impl: str = default_args.model_impl

        calculate_kv_scales: Optional[bool] = default_args.calculate_kv_scales
        additional_config: Optional[Dict[str, Any]] = default_args.additional_config
        enable_reasoning: Optional[bool] = default_args.enable_reasoning
        reasoning_parser: Optional[str] = default_args.reasoning_parser
        use_tqdm_on_load: bool = default_args.use_tqdm_on_load
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
    parser.add_argument("--extra-information",
                        type=str,
                        default=DEFAULT_EXTRA_INFORMATION,
                        help="The path to a json to add to the /info endpoint of the API")
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
    parser.add_argument("--enable-ssl-refresh",
                        action="store_true",
                        default=application_settings.enable_ssl_refresh,
                        help="Refresh SSL Context when SSL certificate files change")
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
    parser.add_argument('--chat-template-content-format',
                        type=str,
                        default=application_settings.chat_template_content_format,
                        choices=get_args(ChatTemplateContentFormatOption),
                        help='The format to render message content within a chat template.'
                        '\n\n'
                        '* "string" will render the content as a string. '
                        'Example: "Hello World"\n'
                        '* "openai" will render the content as a list of dictionaries, '
                        'similar to OpenAI schema. '
                        'Example: [{"type": "text", "text": "Hello world!"}]')
    parser.add_argument("--response-role",
                        type=str,
                        default=application_settings.response_role,
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument("--with-launch-arguments",
                        type=bool,
                        default=application_settings.with_launch_arguments,
                        help="Whether the route launch_arguments should display the launch arguments")
    parser.add_argument("--return-tokens-as-token-ids",
                        default=application_settings.return_tokens_as_token_ids,
                        action=BooleanOptionalAction,
                        help="When --max-logprobs is specified, represents single tokens as "
                        "strings of the form 'token_id:{token_id}' so that tokens that "
                        "are not JSON-encodable can be identified.")
    parser.add_argument('--max-log-len',
                        type=int,
                        default=application_settings.max_log_len,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        '\n\nDefault: Unlimited')
    parser.add_argument("--prompt-adapters",
                        type=nullable_str,
                        default=application_settings.prompt_adapters,
                        nargs='+',
                        action=PromptAdapterParserAction,
                        help="Prompt adapter configurations in the format name=path. "
                        "Multiple adapters can be specified.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        default=application_settings.disable_frontend_multiprocessing,
                        action=BooleanOptionalAction,
                        help="If specified, will run the OpenAI frontend server in the same "
                        "process as the model serving engine.")
    parser.add_argument("--enable-request-id-headers",
                        default=application_settings.enable_request_id_headers,
                        action=BooleanOptionalAction,
                        help="If specified, API server will add X-Request-Id header to "
                        "responses. Caution: this hurts performance at high QPS.")
    parser.add_argument("--enable-auto-tool-choice",
                        default=application_settings.enable_auto_tool_choice,
                        action=BooleanOptionalAction,
                        help=
                        "Enable auto tool choice for supported models. Use --tool-call-parser"
                        "to specify which parser to use")
    valid_reasoning_parsers = ReasoningParserManager.reasoning_parsers.keys()
    valid_tool_parsers = ToolParserManager.tool_parsers.keys()
    parser.add_argument("--tool-call-parser",
                        type=str,
                        metavar="{" + ",".join(valid_tool_parsers) + "} or name registered in "
                        "--tool-parser-plugin",
                        default=application_settings.tool_call_parser,
                        help=
                        "Select the tool call parser depending on the model that you're using."
                        " This is used to parse the model-generated tool call into OpenAI API "
                        "format. Required for --enable-auto-tool-choice.")
    parser.add_argument("--tool-parser-plugin",
                        type=str,
                        default=application_settings.tool_parser_plugin,
                        help=
                        "Special the tool parser plugin write to parse the model-generated tool"
                        " into OpenAI API format, the name register in this plugin can be used "
                        "in --tool-call-parser.")
    parser.add_argument("--disable-fastapi-docs",
                        action='store_true',
                        default=application_settings.disable_fastapi_docs,
                        help="Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint")
    parser.add_argument("--enable-prompt-tokens-details",
                        action='store_true',
                        default=application_settings.enable_prompt_tokens_details,
                        help="If set to True, enable prompt_tokens_details in usage.")
    parser.add_argument("--enable-server-load-tracking",
                        action='store_true',
                        default=application_settings.enable_server_load_tracking,
                        help="If set to True, enable tracking server_load_metrics in the app state.")
    parser.add_argument("--disable-uvicorn-access-log",
                        action="store_true",
                        help="Disable uvicorn access log.")

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
