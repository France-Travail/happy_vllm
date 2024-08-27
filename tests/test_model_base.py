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

from happy_vllm import utils
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
    args = Namespace(model_name=os.environ["MODEL_NAME"], model=os.environ['MODEL'], with_launch_arguments=True)
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