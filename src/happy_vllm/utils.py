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

import logging
from pathlib import Path
import importlib.metadata
from typing import List, Union


logger = logging.getLogger(__name__)

TEST_TOKENIZER_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def get_package_version() -> str:
    '''Returns the current version of the package

    Returns:
        str: version of the package
    '''
    version = importlib.metadata.version("happy_vllm")
    return version


def get_vllm_version() -> str:
    '''Returns the installed version of vLLM 

    Returns:
        str: version of vLLM
    '''
    version = importlib.metadata.version("vllm")
    return version


def proper_tokenization(tokenizer, str_to_tokenize: str) -> tuple:
    """Gets the token ids for a str. We must use this technique in order to get the right token ids
    with their whitespace in the case of some tokenizers (e.g. Llama) adding whitespace in front of the tokenized sentence

    Args:
        tokenizer : A tokenizer
        str_to_tokenize (str) : The string one wants to tokenize

    Returns:
        tuple : The tuple containing the token ids for the input string
    """
    big_word = 'thisisareallybigwordisntit'
    token_ids_big_word = tokenizer(big_word, add_special_tokens=False)['input_ids']
    # We concatenate the big word with the str in order for the str not to be at the beginning of the sentence
    new_text = big_word + str_to_tokenize
    token_ids_new_text = tokenizer(new_text, add_special_tokens=False)['input_ids']
    # We check that part of the str was not "integrated" in a token of the big_word
    while token_ids_big_word != token_ids_new_text[:len(token_ids_big_word)]:
        # If it has been "integrated", we take another big_word
        big_word = big_word[:-1]
        token_ids_big_word = tokenizer(big_word, add_special_tokens=False)['input_ids']
        # We concatenate the big word with the str in order for the str not to be at the beginning of the sentence
        new_text = big_word + str_to_tokenize
        token_ids_new_text = tokenizer(new_text, add_special_tokens=False)['input_ids']
    token_ids_str = tuple(token_ids_new_text[len(token_ids_big_word):])
    return token_ids_str


def proper_decode(tokenizer, token_ids: Union[int, List[int]]) -> str:
    """Gets the corresponding string from the token ids. We must use this technique in order to get the right string
    with their whitespace in the case of some tokenizers (e.g. Llama) deleting whitespace in front of the tokenized sentence
    """
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    extra_token_id = proper_tokenization(tokenizer, 'c')[0]
    token_ids = [extra_token_id] + token_ids
    return tokenizer.decode(token_ids)[1:]