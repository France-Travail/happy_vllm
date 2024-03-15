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
import torch
import shutil
import pytest
import numpy as np
from transformers import AutoTokenizer

from happy_vllm import utils
from happy_vllm.logits_processors.response_pool import VLLMLogitsProcessorResponsePool

from .conftest import TEST_MODELS_DIR


def teardown_module():
    if os.path.exists(TEST_MODELS_DIR):
        if os.path.isdir(TEST_MODELS_DIR):
            shutil.rmtree(TEST_MODELS_DIR)


def test_VLLMLogitsProcessorResponsePool_init():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)

    possible_responses = ['Yes', 'No', "I don't know"]
    logit_processor_response_pool = VLLMLogitsProcessorResponsePool(tokenizer, possible_responses)
    for possible_response in possible_responses:
        assert possible_response in logit_processor_response_pool.possible_tokens_responses
        tokens_in_processor = logit_processor_response_pool.possible_tokens_responses[possible_response]
        direct_tokens = list(utils.proper_tokenization(tokenizer, possible_response))
        assert tokens_in_processor == direct_tokens
        assert logit_processor_response_pool.eos_token_id == tokenizer.eos_token_id


def test_VLLMLogitsProcessorResponsePool_call():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)

    possible_responses = ['Yes', 'No', 'I am not really sure', 'I am not certain']
    logit_processor_response_pool = VLLMLogitsProcessorResponsePool(tokenizer, possible_responses)
    max_token_ids = max([max(tokens_ids) for tokens_ids in logit_processor_response_pool.possible_tokens_responses.values()])
    scores = torch.tensor(range(max_token_ids+1000), dtype=float)
    
    # No input_ids yet
    new_scores = logit_processor_response_pool([], scores)
    first_tokens = {possible_responses_ids[0] for possible_responses_ids in logit_processor_response_pool.possible_tokens_responses.values()}        
    for token in first_tokens:
        assert pytest.approx(scores[token]) == new_scores[token]
    list_other_tokens = [token for token in range(scores.shape[-1]) if token not in first_tokens]
    assert max(new_scores[list_other_tokens]) == -np.inf

    # Began the response 'I am not really sure'
    new_scores = logit_processor_response_pool(logit_processor_response_pool.possible_tokens_responses['I am not really sure'][:-1], scores)
    token_to_test = logit_processor_response_pool.possible_tokens_responses['I am not really sure'][-1]
    assert pytest.approx(scores[token_to_test]) == new_scores[token_to_test]
    list_other_tokens = [token for token in range(scores.shape[-1]) if token!=token_to_test]
    assert max(new_scores[list_other_tokens]) == -np.inf

    # Began the response 'I am not' (same beginning between 'I am not really sure' and 'I am not certain')
    tokens_i_am_not_really_sure = logit_processor_response_pool.possible_tokens_responses['I am not really sure']
    tokens_i_am_not_certain = logit_processor_response_pool.possible_tokens_responses['I am not certain']
    i = 0
    while tokens_i_am_not_really_sure[i] == tokens_i_am_not_certain[i]:
        i += 1
    new_scores = logit_processor_response_pool(tokens_i_am_not_really_sure[:i], scores)
    tokens_to_test = {tokens_i_am_not_really_sure[i], tokens_i_am_not_certain[i]}
    for token in tokens_to_test:
        assert pytest.approx(scores[token]) == new_scores[token]
    list_other_tokens = [token for token in range(scores.shape[-1]) if token not in tokens_to_test]
    assert max(new_scores[list_other_tokens]) == -np.inf

    # The model generated a complete response -> eos-token
    new_scores = logit_processor_response_pool(logit_processor_response_pool.possible_tokens_responses['I am not really sure'], scores)
    eos_token_id = tokenizer.eos_token_id
    assert scores[eos_token_id] == new_scores[eos_token_id]
    list_other_tokens = [token for token in range(scores.shape[-1]) if token!=eos_token_id]
    assert max(new_scores[list_other_tokens]) == -np.inf

    # Empty list
    possible_responses = []
    logit_processor_response_pool = VLLMLogitsProcessorResponsePool(tokenizer, possible_responses)
    scores = torch.tensor(range(1000), dtype=float)
    assert max(scores) == pytest.approx(max(logit_processor_response_pool([], scores)))
    assert min(scores) == pytest.approx(min(logit_processor_response_pool([], scores)))


def test_VLLMLogitsProcessorResponsePool_get_next_possible_tokens():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)

    possible_responses = ['Yes', 'No', 'I am not really sure', 'I am not certain']
    logit_processor_response_pool = VLLMLogitsProcessorResponsePool(tokenizer, possible_responses)

    # No input_ids yet
    next_tokens = logit_processor_response_pool._get_next_possible_tokens([])
    first_tokens = {possible_responses_ids[0] for possible_responses_ids in logit_processor_response_pool.possible_tokens_responses.values()}        
    assert first_tokens == set(next_tokens)

    # Began the response 'I am not really sure'
    next_tokens = logit_processor_response_pool._get_next_possible_tokens(logit_processor_response_pool.possible_tokens_responses['I am not really sure'][:-1])
    token_to_test = logit_processor_response_pool.possible_tokens_responses['I am not really sure'][-1]       
    assert {token_to_test} == set(next_tokens)

    # Began the response 'I am not' (same beginning between 'I am not really sure' and 'I am not certain')
    tokens_i_am_not_really_sure = logit_processor_response_pool.possible_tokens_responses['I am not really sure']
    tokens_i_am_not_certain = logit_processor_response_pool.possible_tokens_responses['I am not certain']
    i = 0
    while tokens_i_am_not_really_sure[i] == tokens_i_am_not_certain[i]:
        i += 1
    next_tokens = logit_processor_response_pool._get_next_possible_tokens(tokens_i_am_not_really_sure[:i])
    tokens_to_test = {tokens_i_am_not_really_sure[i], tokens_i_am_not_certain[i]}
    assert set(next_tokens) == tokens_to_test

    # The model generated a complete response -> eos-token
    next_tokens = logit_processor_response_pool._get_next_possible_tokens(logit_processor_response_pool.possible_tokens_responses['I am not really sure'])
    assert set(next_tokens) == {tokenizer.eos_token_id}


def test_VLLMLogitsProcessorResponsePool_get_common_tokens_ids_end_begin():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)

    possible_responses = ['Yes', 'No', 'I am not really sure', 'I am not certain']
    logit_processor_response_pool = VLLMLogitsProcessorResponsePool(tokenizer, possible_responses)

    list_end = [1, 2, 3, 4]
    list_begin = [3, 4, 5, 6]
    assert 2 == logit_processor_response_pool._get_common_tokens_ids_end_begin(list_end, list_begin)

    list_end = [1, 2, 3, 4, 2]
    list_begin = [3, 4, 5, 6]
    assert 0 == logit_processor_response_pool._get_common_tokens_ids_end_begin(list_end, list_begin)

    list_end = [1, 2, 3]
    list_begin = [3, 4, 5, 6]
    assert 1 == logit_processor_response_pool._get_common_tokens_ids_end_begin(list_end, list_begin)

    list_end = [1, 2, 3]
    list_begin = [4, 5, 6]
    assert 0 == logit_processor_response_pool._get_common_tokens_ids_end_begin(list_end, list_begin)

    list_end = list(range(100))
    list_begin = list(range(100))
    assert 100 == logit_processor_response_pool._get_common_tokens_ids_end_begin(list_end, list_begin)