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
import shutil
import importlib.metadata
from transformers import AutoTokenizer

from happy_vllm import utils

from .conftest import TEST_MODELS_DIR


def teardown_module():
    if os.path.exists(TEST_MODELS_DIR):
        if os.path.isdir(TEST_MODELS_DIR):
            shutil.rmtree(TEST_MODELS_DIR)


def test_get_package_version():
    # Nominal case
    version = utils.get_package_version()
    assert version == importlib.metadata.version("happy_vllm")


def test_get_vllm_version():
    # Nominal case
    version = utils.get_vllm_version()
    assert version == importlib.metadata.version("vllm")


def test_proper_tokenization():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)
    for string in ['.', '!', '?', 'Hey', 'hey']:
        assert utils.proper_tokenization(tokenizer, f' {string}') != utils.proper_tokenization(tokenizer, f'{string}')
    for string in ['.', '!', '?']:
        assert utils.proper_tokenization(tokenizer, f'reallybigword{string}')[-1] == utils.proper_tokenization(tokenizer, f'{string}')[-1]
        assert utils.proper_tokenization(tokenizer, f'reallybigword {string}')[-1] == utils.proper_tokenization(tokenizer, f' {string}')[-1]

def test_proper_decode():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)
    example = "This is an example. It is very important"
    for i in range(len(example)):
        truncated_example = example[i:]
        token_ids = list(utils.proper_tokenization(tokenizer, truncated_example))
        assert truncated_example == utils.proper_decode(tokenizer, token_ids)