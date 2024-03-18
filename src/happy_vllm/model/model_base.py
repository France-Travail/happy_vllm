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


"""This module contains the base Model class
"""


import logging
from pathlib import Path
from argparse import Namespace
from transformers import AutoTokenizer
from typing import Any, Tuple, Union, List
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.transformers_utils.tokenizer import TokenizerGroup
from pydantic_settings import BaseSettings, SettingsConfigDict
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

from happy_vllm import utils

logger = logging.getLogger(__name__)


class ModelSettings(BaseSettings):
    """Download settings

    This class is used for settings management purpose, have a look at the pydantic
    documentation for more details : https://pydantic-docs.helpmanual.io/usage/settings/

    By default, it looks for environment variables (case insensitive) to set the settings
    if a variable is not found, it looks for a file name .env in your working directory
    where you can declare the values of the variables and finally it sets the values
    to the default ones one can define above
    """
    model : str = 'None'
    model_name : str = '?'
    tokenizer_name : str = 'None'
    TEST_MODELS_DIR : str = 'None'

    model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))


class Model:
    """Parent model class.
    """

    def __init__(self, **kwargs):
        '''Init. model class'''
        self._model = None
        self._tokenizer = None
        self._model_conf = None
        self._model_explainer = None
        self._loaded = False

    def is_model_loaded(self):
        """return the state of the model"""
        return self._loaded

    def loading(self, cli_args: Namespace, **kwargs):
        """load the model"""
        self._load_model(cli_args, **kwargs)
        self._loaded = True

    def _load_model(self, cli_args: Namespace, **kwargs) -> None:
        """Load a model from a file

        Returns:
            Tuple[Any, dict]: A tuple containing the model and a dict of metadata about it.
        """
        settings = ModelSettings(**kwargs)
        if settings.model != 'None':
            cli_args.model = settings.model
        if settings.model_name != '?':
            model_name = settings.model_name
        else:
            model_name = cli_args.model_name

        self._model_conf = {'model_name': model_name}

        logger.info(f"Loading the model from {cli_args.model}")
        if model_name != "TEST MODEL":
            engine_args = AsyncEngineArgs.from_cli_args(cli_args) 
            self._model = AsyncLLMEngine.from_engine_args(engine_args) # type: ignore
            if isinstance(self._model.engine.tokenizer, TokenizerGroup): # type: ignore
                self._tokenizer = self._model.engine.tokenizer.tokenizer # type: ignore
            else:
                self._tokenizer = self._model.engine.tokenizer # type: ignore
            self._tokenizer_lmformatenforcer = build_token_enforcer_tokenizer_data(self._tokenizer)
            self.max_model_len = self._model.engine.model_config.max_model_len # type: ignore
            self.original_truncation_side = self._tokenizer.truncation_side
        # For test purpose
        else:
            self.max_model_len = 2048
            self.original_truncation_side = 'right'
            self._tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name,
                                                     cache_dir=settings.TEST_MODELS_DIR, truncation_side=self.original_truncation_side)
            self._tokenizer_lmformatenforcer = build_token_enforcer_tokenizer_data(self._tokenizer)
            self._model = MockModel(self._tokenizer)
        logger.info(f"Model loaded")

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes a text
        
        Args:
            text (str) : The text to tokenize

        Returns:
            list : The list of token ids
        """
        return list(utils.proper_tokenization(self._tokenizer, text))

    def split_text(self, text: str, num_tokens_in_chunk: int = 200, separators: Union[list, None] = None) -> List[str]:
        '''Splits a text in small texts containing at least num_tokens_in_chunk tokens and ending by a separator. note that the `separators` 
        used are the tokenization of the strings and not the strings themselves (which explains why we must for example 
        specify ' .' and '.' as two separate separators) 

        Args:
            text (str) : The text to split

        Kwargs:
            num_tokens_in_chunk (int) : The minimal number of tokens in the chunk
            separators (list) : The separators marking the end of a sentence

        Returns:
            A list of strings each string containing at least num_tokens_in_chunk tokens and ending by a separator
        '''
        if separators is None:
            separators = [".", "!", "?", "|", " .", " !", " ?", " |"]
        separators_tokens_ids = set()
        for separator in separators:
            separators_tokens_ids.add(utils.proper_tokenization(self._tokenizer, separator))
        tokens = list(utils.proper_tokenization(self._tokenizer, text))
        indices_separators = []
        for separator_tokens_ids in separators_tokens_ids:
            indices_separators += find_indices_sub_list_in_list(tokens, list(separator_tokens_ids))
        indices_separators.sort()

        chunks = []
        index_beginning_chunk = 0
        current_used_separator = 0
        while current_used_separator < len(indices_separators):
            index_current_used_separator = indices_separators[current_used_separator]
            if index_current_used_separator +1 - index_beginning_chunk >= num_tokens_in_chunk:
                chunks.append(tokens[index_beginning_chunk:index_current_used_separator + 1])
                index_beginning_chunk = index_current_used_separator + 1
            current_used_separator += 1
        chunks.append(tokens[index_beginning_chunk:])
        chunks = [utils.proper_decode(self._tokenizer, chunk) for chunk in chunks]
        chunks = [element for element in chunks if element!= ""]
        return chunks

    def extract_text_outside_truncation(self, text: str, truncation_side: Union[str, None] = None, max_length: Union[int, None] = None) -> str:
        """Extracts the part of the prompt not kept after truncation, which will not be infered by the model.
        First, we tokenize the prompt while applying truncation.
        We obtain a list of sequences of token ids padded, which are outside the truncation.
        Then we decode this list of tensors of token IDs containing special tokens to a string.

        Args:
            prompt (str)

        Returns:
            prompt outside the truncation (str)
        """
        if max_length is None:
            max_length = self.max_model_len
        if truncation_side is None:
            truncation_side = self.original_truncation_side
        self._tokenizer.truncation_side = truncation_side
        list_tokens = self._tokenizer(text, truncation=True, add_special_tokens=False, max_length=max_length, return_overflowing_tokens=True)['input_ids']
        if len(list_tokens) <= 1:
            return ''
        not_truncated = list_tokens[0]
        truncated_tmp = list_tokens[1:]
        if self._tokenizer.truncation_side == 'left':
            truncated_tmp.reverse()
        truncated = []
        for truncated_tokens in truncated_tmp:
            truncated += truncated_tokens
        truncated_str = self._tokenizer.decode(truncated)
        self._tokenizer.truncation_side = self.original_truncation_side
        return truncated_str

    def get_gpu_kv_cache_usage(self) -> float:
        """Gets the GPU KV cache usage

        Returns:
            The GPU KV cache usage
        """
        total_num_gpu_blocks = self._model.engine.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self._model.engine.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks
        return gpu_cache_usage

    def get_cpu_kv_cache_usage(self) -> float:
        """Gets the CPU KV cache usage

        Returns:
            The CPU KV cache usage
        """
        total_num_cpu_blocks = self._model.engine.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (
                self._model.engine.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0
        return cpu_cache_usage

    def get_status_requests(self) -> dict:
        """Gets the status of the different requests being processed

        Returns:
            A dictionary containing the number of requests in the different status (running, swapped and pending)
        """
        status_requests = {"requests_running": len(self._model.engine.scheduler.running),
                            "requests_swapped": len(self._model.engine.scheduler.swapped),
                            "requests_pending": len(self._model.engine.scheduler.waiting)}
        return status_requests


def find_indices_sub_list_in_list(big_list: list, sub_list: list) -> list:
    """Find the indices of the presence of a sub list in a bigger list. For example
    if big_list = [3, 4, 1, 2, 3, 4, 5, 6, 3, 4] and sub_list = [3, 4],
    the result will be [1, 5, 9]

    Args:
        big_list (list) : The list in which we want to find the sub_list
        sub_list (list): The list we want the indices of in the big_list

    Returns:
        list : The list of indices of where the sub_list is in the big_list 
    """
    len_sub_list = len(sub_list)
    indices = []
    for index in range(len(big_list)):
        if big_list[index - len_sub_list + 1: index + 1] == sub_list:
            indices.append(index)
    return indices


class MockModel():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    async def generate(self, prompt, sampling_params, request_id):
        stream_txts = [f"n={i} "*i + prompt + " This is the generated text. I find it really good don't you ?" for i in range(sampling_params.n)]
        stream_ids = [self.tokenizer(stream_txt, add_special_tokens=False, truncation=True, max_length=sampling_params.max_tokens)['input_ids'] for stream_txt in stream_txts]
        max_length = max([len(element) for element in stream_ids]) + 1
        stream_tmp = [[self.tokenizer.decode(text[:i]) for text in stream_ids] for i in range(max_length)]
        print(stream_tmp)
        stream = [MockGenerateResponse(prompt, texts) for texts in stream_tmp]
        # Mock the length finish_reason
        for i in range(sampling_params.n):
            if stream[-1].outputs[i].finish_reason is None:
                stream[-1].outputs[i].finish_reason = "length"
        stream = self.async_iter(stream)
        async for outputs in stream:
            yield outputs

    async def async_iter(self, my_list):
        for element in my_list:
            yield element


class MockGenerateResponse():

    def __init__(self, prompt, outputs):
        self.prompt = prompt
        outputs = [MockOutput(output) for output in outputs]
        self.outputs = outputs


class MockOutput():
    
    def __init__(self, text):
        self.text = text
        if text[-len("don't you ?"):] == "don't you ?":
            self.finish_reason = "stop"
        else:
            self.finish_reason = None