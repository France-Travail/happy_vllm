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

import os
import pytest
import shutil
from argparse import Namespace
from transformers import AutoTokenizer

from vllm.config import ConfigFormat
from vllm.engine.arg_utils import AsyncEngineArgs

from happy_vllm import utils
from happy_vllm import utils_args
from happy_vllm.model import model_base
from happy_vllm.model.model_base import Model
from happy_vllm.launch import happy_vllm_build_async_engine_client


from .conftest import TEST_MODELS_DIR


def teardown_module():
    if os.path.exists(TEST_MODELS_DIR):
        if os.path.isdir(TEST_MODELS_DIR):
            shutil.rmtree(TEST_MODELS_DIR)


def init_model(truncation_side="left"):
    model = Model(app_name=os.environ['app_name'])
    model._tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR, truncation_side=truncation_side)
    model.original_truncation_side = truncation_side
    model.max_model_len = 2048
    return model


@pytest.mark.asyncio
async def test_is_model_loaded():
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
        with_launch_arguments=False,
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
        speculative_disable_mqa_scorer=False,
        scheduling_policy="fcfs",
        config_format=ConfigFormat.AUTO,
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
    model = init_model()
    assert not(model.is_model_loaded())
    await model.loading(happy_vllm_build_async_engine_client(args), args)
    assert model.is_model_loaded()


def test_tokenize():
    model = init_model()
    prompts = ["Hey it's me. How are you ?", "Fine. You ?", " I'm feeling really great "]
    for prompt in prompts:
        result = model.tokenize(prompt)
        target = list(utils.proper_tokenization(model._tokenizer, prompt))
        assert result == target


def test_split_text():
    model = init_model()
    text = "Hey, my name is LLM. How are you ? Fine, you ? That's wonderful news : I'm also fine. But do you think it will last ?"
        
    # Nominal, all split
    num_tokens_in_chunk = 2
    target_split_text = ["Hey, my name is LLM.",
                        " How are you ?", 
                        " Fine, you ?",
                        " That's wonderful news : I'm also fine.",
                        " But do you think it will last ?"]
    split_text = model.split_text(text, num_tokens_in_chunk)
    assert split_text == target_split_text

    # Nominal, some split
    num_tokens_in_chunk = 6
    target_split_text = ["Hey, my name is LLM.",
                        " How are you ? Fine, you ?",
                        " That's wonderful news : I'm also fine.",
                        " But do you think it will last ?"]
    split_text = model.split_text(text, num_tokens_in_chunk)
    assert split_text == target_split_text

    # Nominal, no split
    num_tokens_in_chunk = 1000
    target_split_text = [text]
    split_text = model.split_text(text, num_tokens_in_chunk)
    assert split_text == target_split_text

    # Change separator tokens
    num_tokens_in_chunk = 2
    separators = [" ?"]
    target_split_text = ["Hey, my name is LLM. How are you ?",
                        " Fine, you ?",
                        " That's wonderful news : I'm also fine. But do you think it will last ?"]
    split_text = model.split_text(text, num_tokens_in_chunk, separators=separators)
    assert split_text == target_split_text

    # No separator
    num_tokens_in_chunk = 2
    separators = []
    target_split_text = [text]
    split_text = model.split_text(text, num_tokens_in_chunk, separators=separators)
    assert split_text == target_split_text

    # Separator not present in the text
    num_tokens_in_chunk = 2
    new_text = "Unfortunately, i haven't put any separators in this sentence; I wonder what will happen ;"
    target_split_text = [new_text]
    split_text = model.split_text(new_text, num_tokens_in_chunk, separators=separators)
    assert split_text == target_split_text


def test_extract_text_outside_truncation():
    model = init_model()
    max_length = 4

    prompt = "Hey, it's me. How are you ?"

    # Left truncation
    truncation_side = "left"
    token_ids = model._tokenizer(prompt, add_special_tokens=False)['input_ids']
    target_truncated_prompt = model._tokenizer.decode(token_ids[:-max_length])
    truncated_prompt = model.extract_text_outside_truncation(prompt, truncation_side, max_length)
    assert truncated_prompt == target_truncated_prompt

    # Right truncation
    truncation_side = "right"
    token_ids = model._tokenizer(prompt, add_special_tokens=False)['input_ids']
    target_truncated_prompt = model._tokenizer.decode(token_ids[max_length:])
    truncated_prompt = model.extract_text_outside_truncation(prompt, truncation_side, max_length)
    assert truncated_prompt == target_truncated_prompt

    # No truncation
    max_length = 100
    truncation_side = "left"
    target_truncated_prompt = ""
    truncated_prompt = model.extract_text_outside_truncation(prompt, truncation_side, max_length)
    assert truncated_prompt == target_truncated_prompt


def test_find_indices_sub_list_in_list():
    # Nominal case
    big_list = [3, 4, 1, 2, 3, 4, 5, 6, 3, 4]
    sub_list = [3, 4]
    target_indices = [1, 5, 9]
    assert model_base.find_indices_sub_list_in_list(big_list, sub_list) == target_indices

    # Presence of the beginning of sub_list
    big_list = [3, 4, 1, 2, 3, 4, 5, 6, 3]
    sub_list = [3, 4]
    target_indices = [1, 5]
    assert model_base.find_indices_sub_list_in_list(big_list, sub_list) == target_indices

    big_list = [3, 4, 1, 2, 3, 5, 6, 3, 4]
    sub_list = [3, 4]
    target_indices = [1, 8]
    assert model_base.find_indices_sub_list_in_list(big_list, sub_list) == target_indices

    # Presence the end of sub_list
    big_list = [3, 4, 1, 2, 4, 5, 6, 3, 4]
    sub_list = [3, 4]
    target_indices = [1, 8]
    assert model_base.find_indices_sub_list_in_list(big_list, sub_list) == target_indices

    big_list = [3, 4, 1, 2, 3, 4, 5, 6, 4]
    sub_list = [3, 4]
    target_indices = [1, 5]
    assert model_base.find_indices_sub_list_in_list(big_list, sub_list) == target_indices