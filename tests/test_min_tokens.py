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
import math
import torch
import shutil
import pytest
import numpy as np
from transformers import AutoTokenizer

from happy_vllm import utils
from happy_vllm.logits_processors.min_tokens import VLLMLogitsProcessorMinTokens

from .conftest import TEST_MODELS_DIR


def teardown_module():
    if os.path.exists(TEST_MODELS_DIR):
        if os.path.isdir(TEST_MODELS_DIR):
            shutil.rmtree(TEST_MODELS_DIR)


def test_VLLMLogitsProcessorMinTokens_init():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)

    list_min_tokens = [10, 20, 30, 50]
    for min_tokens in list_min_tokens:
        logit_processor_min_tokens = VLLMLogitsProcessorMinTokens(tokenizer, min_tokens)
        assert logit_processor_min_tokens.min_tokens == min_tokens
        assert logit_processor_min_tokens.eos_token_id == tokenizer.eos_token_id


def test_VLLMLogitsProcessorMinTokens_call():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)

    logit_processor_min_tokens = VLLMLogitsProcessorMinTokens(tokenizer, 10)
    scores = torch.tensor([10] * max(tokenizer.eos_token_id + 10, 1000), dtype=float)

    for input_ids in [list(range(5, i)) for i in range(5, 30)]:
        new_scores = logit_processor_min_tokens(input_ids, scores)
        if len(input_ids) < 10:
            assert new_scores[tokenizer.eos_token_id] == -math.inf
        else:
            assert new_scores[tokenizer.eos_token_id] == 10