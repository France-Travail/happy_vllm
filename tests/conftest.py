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


"""Test configuration

This conftest.py file is the first loaded by pytest when the tests are executed,
see the documentation for more infos :
https://docs.pytest.org/en/7.2.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files

We use it to :
- create a test model
- create a AsyncClient named test_base_client which does not load the test model
- create a AsyncClient named test_complete_client which does load the test model
- set some environment variables to check if they are well set in the app settings

> More details about test_base_client and test_complete_client:
>
> By default a AsyncClient does not triggered lifespan so the startup and shutdown events are never fired
> and the model is never loaded.
> We can fire thoses events by using AsyncClient as a context manager so we use two AsyncClient in
> our tests : a test_base_client that does not load the model and test_complete_client that does
> load the model thanks to a context manager. It allows us to test the behavior of our application
> when a model is not loaded (which should not happen).
>
> AsyncClient need to use ASGITransport to be sure the routes are available and avoid 404 response
> As AsyncClient don't trigger the lifespan by default in httpx, we have to use asgi_lifespan package
> (source httpx.AsyncClient doc) to init a LifespanManager and trigger the lifespan
"""

import os
import pytest
import pytest_asyncio
from typing import Union
from pathlib import Path
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport

from argparse import Namespace
from pydantic_settings import BaseSettings, SettingsConfigDict


# Manage the huggingface token
class HuggingfaceSettings(BaseSettings):
    """A class to get the HuggingFace token
    """
    hf_token : Union[str, None] = None
    model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))

huggingface_settings = HuggingfaceSettings()
os.environ['HF_TOKEN'] = huggingface_settings.hf_token

# Set paths
TEST_DIR = Path(__file__).parent.resolve()
TEST_MODELS_DIR = TEST_DIR / "data" / "models"

# Set environment variables for testing
os.environ["app_name"] = "APP_TESTS"
os.environ["api_endpoint_prefix"] = "/tests"
os.environ["MODEL_NAME"] = "TEST MODEL"
os.environ["MODEL"] = "test"
os.environ["TEST_MODELS_DIR"] = str(TEST_MODELS_DIR)
os.environ["TEST_MODE"] = str(True)

# We must import the utils module after setting the environnement variables because
# it also imports the .core folder via the __init__ and it may impact the other tests
from happy_vllm import utils
os.environ["tokenizer_name"] = utils.TEST_TOKENIZER_NAME

from happy_vllm.core import resources
from happy_vllm.model.model_base import Model
from happy_vllm.application import declare_application
from happy_vllm.launch import happy_vllm_build_async_engine_client


@pytest_asyncio.fixture(scope="session")
async def test_base_client() -> AsyncClient:
    """Basic AsyncClient that do not run startup and shutdown events"""
    args = Namespace(
        model_name=os.environ['MODEL_NAME'],
        model=os.environ['MODEL'],
        app_name=os.environ["app_name"],
        api_endpoint_prefix=os.environ["api_endpoint_prefix"],
        allow_credentials=True,
        allowed_origins=["*"],
        allowed_methods=["*"],
        allowed_headers=["*"],
        explicit_errors=False,
        uvicorn_log_level='info',
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        root_path=None,
        lora_modules=None,
        chat_template=None,
        response_role='assistant',
        with_launch_arguments=True,
        max_log_len=None,
        prompt_adapters=None,
        return_tokens_as_token_ids=False,
        disable_frontend_multiprocessing=False,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        tokenizer=None,
        skip_tokenizer_init=False,
        revision=None,
        code_revision=None,
        tokenizer_revision=None,
        tokenizer_mode='auto',
        trust_remote_code=False,
        download_dir=None,
        load_format='auto',
        config_format='auto',
        dtype='auto',
        kv_cache_dtype='auto',
        quantization_param_path=None,
        max_model_len=None,
        guided_decoding_backend='outlines',
        distributed_executor_backend=None,
        worker_use_ray=False,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        max_parallel_loading_workers=None,
        ray_workers_use_nsight=False,
        block_size=16,
        enable_prefix_caching=False,
        disable_sliding_window=False,
        use_v2_block_manager=False,
        num_lookahead_slots=0,
        seed=0,
        swap_space=4.0,
        cpu_offload_gb=0.0,
        gpu_memory_utilization=0.9,
        num_gpu_blocks_override=None,
        max_num_batched_tokens=None,
        max_num_seqs=256,
        max_logprobs=20,
        disable_log_stats=False,
        quantization=None,
        rope_scaling=None,
        rope_theta=None,
        enforce_eager=False,
        max_context_len_to_capture=None,
        max_seq_len_to_capture=8192,
        disable_custom_all_reduce=False,
        tokenizer_pool_size=0,
        tokenizer_pool_type='ray',
        tokenizer_pool_extra_config=None,
        limit_mm_per_prompt=None,
        mm_processor_kwargs=None,
        enable_lora=False,
        max_loras=1,
        max_lora_rank=16,
        lora_extra_vocab_size=256,
        lora_dtype='auto',
        long_lora_scaling_factors=None,
        max_cpu_loras=None,
        fully_sharded_loras=False,
        enable_prompt_adapter=False,
        max_prompt_adapters=1,
        max_prompt_adapter_token=0,
        device='auto',
        num_scheduler_steps=1,
        multi_step_stream_outputs=False,
        scheduler_delay_factor=0.0,
        enable_chunked_prefill=None,
        speculative_model=None,
        speculative_model_quantization=None,
        num_speculative_tokens=None,
        speculative_draft_tensor_parallel_size=None,
        speculative_max_model_len=None,
        speculative_disable_by_batch_size=None,
        ngram_prompt_lookup_max=None,
        ngram_prompt_lookup_min=None,
        spec_decoding_acceptance_method='rejection_sampler',
        typical_acceptance_sampler_posterior_threshold=None,
        typical_acceptance_sampler_posterior_alpha=None,
        disable_logprobs_during_spec_decoding=None,
        model_loader_extra_config=None,
        ignore_patterns=[],
        preemption_mode=None,
        served_model_name=None,
        qlora_adapter_name_or_path=None,
        otlp_traces_endpoint=None,
        collect_detailed_traces=None,
        disable_async_output_proc=False,
        override_neuron_config=None,
        disable_log_requests=False,
        engine_use_ray=False
    )
    app = await declare_application(happy_vllm_build_async_engine_client(args), args)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True)


@pytest_asyncio.fixture()
async def test_complete_client(monkeypatch) -> AsyncClient:
    """Complete AsyncClient that do run startup and shutdown events to load
    the model
    """
    # Use base model for tests
    monkeypatch.setattr(resources, "Model", Model)
    args = Namespace(
        model_name=os.environ['MODEL_NAME'],
        model=os.environ['MODEL'],
        app_name=os.environ["app_name"],
        api_endpoint_prefix=os.environ["api_endpoint_prefix"],
        allow_credentials=True,
        allowed_origins=["*"],
        allowed_methods=["*"],
        allowed_headers=["*"],
        explicit_errors=False,
        uvicorn_log_level='info',
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        root_path=None,
        lora_modules=None,
        chat_template=None,
        response_role='assistant',
        with_launch_arguments=True,
        max_log_len=None,
        prompt_adapters=None,
        return_tokens_as_token_ids=False,
        disable_frontend_multiprocessing=False,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        tokenizer=None,
        skip_tokenizer_init=False,
        revision=None,
        code_revision=None,
        tokenizer_revision=None,
        tokenizer_mode='auto',
        trust_remote_code=False,
        download_dir=None,
        load_format='auto',
        config_format='auto',
        dtype='auto',
        kv_cache_dtype='auto',
        quantization_param_path=None,
        max_model_len=None,
        guided_decoding_backend='outlines',
        distributed_executor_backend=None,
        worker_use_ray=False,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        max_parallel_loading_workers=None,
        ray_workers_use_nsight=False,
        block_size=16,
        enable_prefix_caching=False,
        disable_sliding_window=False,
        use_v2_block_manager=False,
        num_lookahead_slots=0,
        seed=0,
        swap_space=4.0,
        cpu_offload_gb=0.0,
        gpu_memory_utilization=0.9,
        num_gpu_blocks_override=None,
        max_num_batched_tokens=None,
        max_num_seqs=256,
        max_logprobs=20,
        disable_log_stats=False,
        quantization=None,
        rope_scaling=None,
        rope_theta=None,
        enforce_eager=False,
        max_context_len_to_capture=None,
        max_seq_len_to_capture=8192,
        disable_custom_all_reduce=False,
        tokenizer_pool_size=0,
        tokenizer_pool_type='ray',
        tokenizer_pool_extra_config=None,
        limit_mm_per_prompt=None,
        mm_processor_kwargs=None,
        enable_lora=False,
        max_loras=1,
        max_lora_rank=16,
        lora_extra_vocab_size=256,
        lora_dtype='auto',
        long_lora_scaling_factors=None,
        max_cpu_loras=None,
        fully_sharded_loras=False,
        enable_prompt_adapter=False,
        max_prompt_adapters=1,
        max_prompt_adapter_token=0,
        device='auto',
        num_scheduler_steps=1,
        multi_step_stream_outputs=False,
        scheduler_delay_factor=0.0,
        enable_chunked_prefill=None,
        speculative_model=None,
        speculative_model_quantization=None,
        num_speculative_tokens=None,
        speculative_draft_tensor_parallel_size=None,
        speculative_max_model_len=None,
        speculative_disable_by_batch_size=None,
        ngram_prompt_lookup_max=None,
        ngram_prompt_lookup_min=None,
        spec_decoding_acceptance_method='rejection_sampler',
        typical_acceptance_sampler_posterior_threshold=None,
        typical_acceptance_sampler_posterior_alpha=None,
        disable_logprobs_during_spec_decoding=None,
        model_loader_extra_config=None,
        ignore_patterns=[],
        preemption_mode=None,
        served_model_name=None,
        qlora_adapter_name_or_path=None,
        otlp_traces_endpoint=None,
        collect_detailed_traces=None,
        disable_async_output_proc=False,
        override_neuron_config=None,
        disable_log_requests=False,
        engine_use_ray=False
    )
    app = await declare_application(happy_vllm_build_async_engine_client(args), args)
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test", follow_redirects=True) as client:
            print(f"RESSOURCE : {resources.RESOURCES}")
            yield client