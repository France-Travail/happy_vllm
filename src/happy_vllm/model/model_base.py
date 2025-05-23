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


import os
import json
import asyncio
import logging
from pathlib import Path
from argparse import Namespace
from transformers import AutoTokenizer
from typing import Any, Tuple, Union, List, cast
from vllm.entrypoints.logger import RequestLogger
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.protocol import TokenizeResponse, DetokenizeResponse
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.transformers_utils.tokenizer_group.tokenizer_group import TokenizerGroup
from vllm.entrypoints.openai.serving_transcription import OpenAIServingTranscription
from vllm.entrypoints.openai.serving_models import (BaseModelPath, OpenAIServingModels)

from happy_vllm import utils


logger = logging.getLogger(__name__)


class Model:
    """Parent model class.
    """

    def __init__(self, **kwargs):
        '''Init. model class'''
        self._model = None
        self._tokenizer = None
        self._model_conf = None
        self._model_explainer = None
        self.openai_serving_chat = None
        self.openai_serving_embedding = None
        self.openai_serving_completion = None
        self.openai_serving_tokenization = None
        self._loaded = False
        self.app_name = kwargs.get('app_name', "happy_vllm")
        self.extra_information = {}

    def is_model_loaded(self):
        """return the state of the model"""
        return self._loaded

    async def loading(self, async_engine_client: MQLLMEngineClient, args: Namespace, **kwargs):
        """load the model"""
        await self._load_model(async_engine_client, args, **kwargs)
        self._loaded = True
        if args.extra_information:
            with open(args.extra_information, 'r') as json_file:
                self.extra_information = json.load(json_file)
        if args.with_launch_arguments:
            self.launch_arguments = vars(args)
        else:
            self.launch_arguments = {}

    async def _load_model(self, async_engine_client: MQLLMEngineClient, args: Namespace, **kwargs) -> None:
        """Load a model from a file

        Returns:
            Tuple[Any, dict]: A tuple containing the model and a dict of metadata about it.
        """

        self._model_conf = {'model_name': args.model_name}
    
        logger.info(f"Loading the model from {args.model}")
        if args.model_name != "TEST MODEL":
            self._model = async_engine_client
            model_config = await self._model.get_model_config()
            # Define the tokenizer differently if we have an AsyncLLMEngine
            if isinstance(self._model, AsyncLLMEngine):
                tokenizer_tmp = self._model.engine.tokenizer
            else:
                tokenizer_tmp = self._model.tokenizer
            if isinstance(tokenizer_tmp, TokenizerGroup): # type: ignore
                self._tokenizer = tokenizer_tmp.tokenizer # type: ignore
            else:
                self._tokenizer = tokenizer_tmp # type: ignore
            self.max_model_len = model_config.max_model_len # type: ignore
            # To take into account Mistral tokenizers
            try:
                self.original_truncation_side = self._tokenizer.truncation_side # type: ignore
            except:
                self.original_truncation_side = "left"
            if args.disable_log_requests:
                request_logger = None
            else:
                request_logger = RequestLogger(max_log_len=args.max_log_len)
            served_model_names = [args.model_name]
            base_model_paths = [
                BaseModelPath(name=name, model_path=args.model)
                for name in served_model_names
            ]
            resolved_chat_template = load_chat_template(args.chat_template)
            logger.info("Using supplied chat template:\n%s", resolved_chat_template)
            self.openai_serving_models = OpenAIServingModels(
                                            engine_client=cast(AsyncLLMEngine,self._model),
                                            model_config=model_config,
                                            base_model_paths=base_model_paths,
                                            lora_modules=args.lora_modules,
                                            prompt_adapters=args.prompt_adapters,
                                        )
            self.task = model_config.task
            if model_config.task == "generate":
                self.openai_serving_chat = OpenAIServingChat(cast(AsyncLLMEngine,self._model), model_config,
                                                            self.openai_serving_models,
                                                            args.response_role,
                                                            request_logger=request_logger,
                                                            chat_template=resolved_chat_template,
                                                            chat_template_content_format=args.chat_template_content_format,
                                                            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
                                                            enable_auto_tools=args.enable_auto_tool_choice,
                                                            tool_parser=args.tool_call_parser,
                                                            enable_reasoning=args.enable_reasoning,
                                                            reasoning_parser=args.reasoning_parser,
                                                            enable_prompt_tokens_details=args.enable_prompt_tokens_details)
                self.openai_serving_completion = OpenAIServingCompletion(cast(AsyncLLMEngine,self._model), model_config, 
                                                                        self.openai_serving_models, 
                                                                        request_logger=request_logger,
                                                                        return_tokens_as_token_ids=args.return_tokens_as_token_ids)
            if model_config.task == "embed":
                self.openai_serving_embedding = OpenAIServingEmbedding(cast(AsyncLLMEngine,self._model),
                                                                        model_config,
                                                                        self.openai_serving_models,
                                                                        request_logger=request_logger,
                                                                        chat_template=resolved_chat_template,
                                                                        chat_template_content_format=args.chat_template_content_format)
            if model_config.task == "transcription":
                self.openai_serving_transcription = OpenAIServingTranscription(cast(AsyncLLMEngine,self._model),
                                                                                model_config,
                                                                                self.openai_serving_models,
                                                                                request_logger=request_logger)
            self.openai_serving_tokenization = OpenAIServingTokenization(cast(AsyncLLMEngine,self._model), model_config, 
                                                                        self.openai_serving_models,
                                                                        request_logger=request_logger,
                                                                        chat_template=resolved_chat_template,
                                                                        chat_template_content_format=args.chat_template_content_format)

        # For test purpose
        else:
            self.max_model_len = 2048
            self.original_truncation_side = 'right'
            self._tokenizer = AutoTokenizer.from_pretrained(utils.TEST_TOKENIZER_NAME,
                                                     cache_dir=os.environ["TEST_MODELS_DIR"], truncation_side=self.original_truncation_side)
            self._model = MockModel(self._tokenizer, self.app_name)
            self.openai_serving_tokenization = MockOpenAIServingTokenization(self._tokenizer)
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
        chunks_decoded = [utils.proper_decode(self._tokenizer, chunk) for chunk in chunks]
        chunks_decoded = [element for element in chunks_decoded if element!= ""]
        return chunks_decoded

    def extract_text_outside_truncation(self, text: str, truncation_side: Union[str, None] = None, max_length: Union[int, None] = None) -> str:
        """Extracts the part of the prompt not kept after truncation, which will not be infered by the model.
        First, we tokenize the prompt while applying truncation.
        We obtain a list of sequences of token ids padded, which are outside the truncation.
        Then we decode this list of tensors of token IDs containing special tokens to a string.

        Args:
            text (str) : The text we want to parse
            truncation_side (str) : The side of the truncation
            max_length (int) : The length above which the text will be truncated

        Returns:
            The part of the text which will be dropped by the truncation (str)
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


class MockOpenAIServingTokenization():

    def __init__(self, tokenizer):
        self.tokenizer=tokenizer

    async def create_tokenize(self, request, raw_request):
        token = self.tokenizer(request.prompt, add_special_tokens=request.add_special_tokens)['input_ids']
        return TokenizeResponse(tokens=token,
                                count=len(token),
                                max_model_len=1)

    async def create_detokenize(self, request, raw_request):
        decode = self.tokenizer.decode(request.tokens)
        return DetokenizeResponse(prompt=decode)


class MockModel():

    def __init__(self, tokenizer, app_name: str = "happy_vllm"):
        self.tokenizer = tokenizer
        self.app_name = app_name

    async def async_iter(self, my_list):
        for element in my_list:
            yield element

    async def do_log_stats(self):
        pass


class MockGenerateResponse():

    def __init__(self, prompt, outputs, tokenizer):
        self.prompt = prompt
        outputs = [MockOutput(output, tokenizer) for output in outputs]
        self.outputs = outputs
        self.prompt_token_ids = tokenizer(self.prompt)['input_ids']


class MockOutput():
    
    def __init__(self, text, tokenizer):
        self.text = text
        self.token_ids = tokenizer(text)['input_ids']
        if text[-len("don't you ?"):] == "don't you ?":
            self.finish_reason = "stop"
        else:
            self.finish_reason = None


