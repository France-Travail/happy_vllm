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
import pytest

from transformers import AutoTokenizer
from fastapi.testclient import TestClient
from vllm.sampling_params import SamplingParams
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data


from happy_vllm import utils
from .conftest import TEST_MODELS_DIR
from happy_vllm.routers import functional
from happy_vllm.model.model_base import Model
from happy_vllm.logits_processors.json_format import VLLMLogitsProcessorJSON
from happy_vllm.logits_processors.min_tokens import VLLMLogitsProcessorMinTokens
from happy_vllm.logits_processors.response_pool import VLLMLogitsProcessorResponsePool


def teardown_module():
    if os.path.exists(TEST_MODELS_DIR):
        if os.path.isdir(TEST_MODELS_DIR):
            shutil.rmtree(TEST_MODELS_DIR)


def init_model(truncation_side="left"):
    model = Model()
    model._tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR, truncation_side=truncation_side)
    model._tokenizer_lmformatenforcer = build_token_enforcer_tokenizer_data(model._tokenizer)
    model.original_truncation_side = truncation_side
    model.max_model_len = 2048
    return model


def test_parse_logits_processors():
    """Tests the function parse_logits_processors"""
    model = init_model()
    tokenizer = model._tokenizer
    tokenizer_lmformatenforcer = model._tokenizer_lmformatenforcer
    
    # With response_pool
    prompt = "This is the prompt"
    for response_pool in [["Yes", "No"], ["Certainly", "I think so", "Of course"], ["Without a doubt"]]:
        request_dict = {"temperature": 0, "keyword": "value", "response_pool": response_pool}
        logits_processors = functional.parse_logits_processors(request_dict, prompt, model, tokenizer, tokenizer_lmformatenforcer)
        assert request_dict == {"temperature": 0, "keyword": "value"}
        assert len(logits_processors) == 1
        assert isinstance(logits_processors[0], VLLMLogitsProcessorResponsePool)
        assert len(logits_processors[0].possible_tokens_responses) == len(response_pool)
        assert logits_processors[0].eos_token_id == tokenizer.eos_token_id

    # With min_tokens
    prompt = "This is the prompt"
    for min_tokens in [10, 500]:
        request_dict = {'temperature': 0, "keyword": "value", "min_tokens": min_tokens}
        logits_processors = functional.parse_logits_processors(request_dict, prompt, model, tokenizer, tokenizer_lmformatenforcer)
        assert request_dict == {"temperature": 0, "keyword": "value"}
        assert len(logits_processors) == 1
        assert isinstance(logits_processors[0], VLLMLogitsProcessorMinTokens)
        assert logits_processors[0].min_tokens == min_tokens
        assert logits_processors[0].eos_token_id == tokenizer.eos_token_id

    # With json_format
    prompt = "This is the prompt"
    json_format = {"name": "string",
                    "surname": "string",
                    "favorite_food": "string",
                    "current_number_of_children": "integer",
                    "team_name": "string",
                    "job_title": "string"}
    request_dict = {"temperature": 0, "keyword": "value", "json_format": json_format}
    logits_processors = functional.parse_logits_processors(request_dict, prompt, model, tokenizer, tokenizer_lmformatenforcer)
    assert request_dict == {"temperature": 0, "keyword": "value"}
    assert len(logits_processors) == 1
    assert isinstance(logits_processors[0], VLLMLogitsProcessorJSON)
    assert logits_processors[0].logits_processor_activated

    # With json_format
    prompt = "This is the prompt"
    json_format = {}
    request_dict = {"temperature": 0, "keyword": "value", "json_format": json_format}
    logits_processors = functional.parse_logits_processors(request_dict, prompt, model, tokenizer, tokenizer_lmformatenforcer)
    assert request_dict == {"temperature": 0, "keyword": "value"}
    assert len(logits_processors) == 1
    assert isinstance(logits_processors[0], VLLMLogitsProcessorJSON)
    assert not(logits_processors[0].logits_processor_activated)

    # With min_tokens and response_pool
    prompt = "This is the prompt"
    for min_tokens in [10, 500]:
        for response_pool in [["Yes", "No"], ["Certainly", "I think so", "Of course"], ["Without a doubt"]]:
            request_dict = {'temperature': 0, "keyword": "value", "min_tokens": min_tokens, "response_pool": response_pool}
            logits_processors = functional.parse_logits_processors(request_dict, prompt, model, tokenizer, tokenizer_lmformatenforcer)
            assert request_dict == {"temperature": 0, "keyword": "value"}
            assert len(logits_processors) == 2
            if isinstance(logits_processors[0], VLLMLogitsProcessorMinTokens):
                assert isinstance(logits_processors[1], VLLMLogitsProcessorResponsePool)
                assert logits_processors[0].min_tokens == min_tokens
                assert logits_processors[0].eos_token_id == tokenizer.eos_token_id
                assert len(logits_processors[1].possible_tokens_responses) == len(response_pool)
                assert logits_processors[1].eos_token_id == tokenizer.eos_token_id
            else:
                assert isinstance(logits_processors[0], VLLMLogitsProcessorResponsePool)
                assert isinstance(logits_processors[1], VLLMLogitsProcessorMinTokens)
                assert logits_processors[1].min_tokens == min_tokens
                assert logits_processors[1].eos_token_id == tokenizer.eos_token_id
                assert len(logits_processors[0].possible_tokens_responses) == len(response_pool)
                assert logits_processors[0].eos_token_id == tokenizer.eos_token_id
            

    # Without logits_processors
    request_dict = {"temperature": 0, "keyword": "value"}
    logits_processors = functional.parse_logits_processors(request_dict, prompt, model, tokenizer, tokenizer_lmformatenforcer)
    assert request_dict == {"temperature": 0, "keyword": "value"}
    assert logits_processors == []


def test_parse_generate_parameters():
    """Tests the function parse_generate_parameters"""
    model = init_model()
    tokenizer = model._tokenizer
    tokenizer_lmformatenforcer = model._tokenizer_lmformatenforcer
    
    # with response_pool
    prompt_ini = "This is the prompt"
    temperature = 0
    request_dict = {"temperature": temperature, "response_pool": ["Yes", "No"], "prompt": prompt_ini, "max_tokens": 100}
    prompt, prompt_in_response, sampling_params = functional.parse_generate_parameters(request_dict, model, tokenizer, tokenizer_lmformatenforcer)
    assert prompt == prompt_ini
    assert not(prompt_in_response)
    assert isinstance(sampling_params, SamplingParams)
    assert sampling_params.temperature == temperature
    assert len(sampling_params.logits_processors) == 1
    assert isinstance(sampling_params.logits_processors[0], VLLMLogitsProcessorResponsePool)

    # with min_tokens
    prompt_ini = "This is the prompt"
    temperature = 0
    request_dict = {"temperature": temperature, "min_tokens": 10, "prompt": prompt_ini, "max_tokens": 100}
    prompt, prompt_in_response, sampling_params = functional.parse_generate_parameters(request_dict, model, tokenizer, tokenizer_lmformatenforcer)
    assert prompt == prompt_ini
    assert not(prompt_in_response)
    assert isinstance(sampling_params, SamplingParams)
    assert sampling_params.temperature == temperature
    assert len(sampling_params.logits_processors) == 1
    assert isinstance(sampling_params.logits_processors[0], VLLMLogitsProcessorMinTokens)

    # with json_format
    prompt_ini = "This is the prompt"
    temperature = 0
    json_format = {"name": "string",
                    "surname": "string",
                    "favorite_food": "string",
                    "current_number_of_children": "integer",
                    "team_name": "string",
                    "job_title": "string"}
    request_dict = {"temperature": temperature, "json_format": json_format, "prompt": prompt_ini, "max_tokens": 100}
    prompt, prompt_in_response, sampling_params = functional.parse_generate_parameters(request_dict, model, tokenizer, tokenizer_lmformatenforcer)
    assert prompt == prompt_ini
    assert not(prompt_in_response)
    assert isinstance(sampling_params, SamplingParams)
    assert sampling_params.temperature == temperature
    assert len(sampling_params.logits_processors) == 1
    assert isinstance(sampling_params.logits_processors[0], VLLMLogitsProcessorJSON)

    # with no logits_processors and prompt_in_response=True
    prompt_ini = "This is the prompt but it is longer than before"
    temperature = 1.5
    max_tokens = 123
    request_dict = {"temperature": temperature, "prompt": prompt_ini, "max_tokens": max_tokens, "prompt_in_response": True}
    prompt, prompt_in_response, sampling_params = functional.parse_generate_parameters(request_dict, model, tokenizer, tokenizer_lmformatenforcer)
    assert prompt == prompt_ini
    assert prompt_in_response
    assert isinstance(sampling_params, SamplingParams)
    assert sampling_params.temperature == temperature
    assert sampling_params.max_tokens == max_tokens
    assert sampling_params.logits_processors == []

    # with no logits_processors and prompt_in_response=False
    prompt_ini = "Here"
    temperature = 0.1
    max_tokens = 1234
    request_dict = {"temperature": temperature, "prompt": prompt_ini, "max_tokens": max_tokens, "prompt_in_response": False}
    prompt, prompt_in_response, sampling_params = functional.parse_generate_parameters(request_dict, model, tokenizer, tokenizer_lmformatenforcer)
    assert prompt == prompt_ini
    assert not(prompt_in_response)
    assert isinstance(sampling_params, SamplingParams)
    assert sampling_params.temperature == temperature
    assert sampling_params.max_tokens == max_tokens
    assert sampling_params.logits_processors == []

    # Raise ValueError if min_tokens superior to max_tokens
    prompt_ini = "Here"
    temperature = 0.1
    max_tokens = 10
    min_tokens = 20
    request_dict = {'temperature': temperature, "prompt": prompt_ini, "max_tokens": max_tokens, "min_tokens": min_tokens}
    with pytest.raises(ValueError):
        prompt, prompt_in_response, sampling_params = functional.parse_generate_parameters(request_dict, model, tokenizer, tokenizer_lmformatenforcer)

    # Raise ValueError if response_pool and min_tokens are present
    prompt_ini = "This is the prompt"
    temperature = 0
    request_dict = {"temperature": temperature, "response_pool": ["Yes", "No"], "prompt": prompt_ini, "min_tokens": 10, "max_tokens": 100}
    with pytest.raises(ValueError):
        prompt, prompt_in_response, sampling_params = functional.parse_generate_parameters(request_dict, model, tokenizer, tokenizer_lmformatenforcer)


def test_generate(test_complete_client: TestClient):
    """Test the route generate thanks to the test_complete_client we created in conftest.py"""
    model = init_model()
    tokenizer = model._tokenizer

    def get_response(tokenizer, prompt, i, max_tokens):
        prompt_tot = f"n={i} "*i + prompt + " This is the generated text. I find it really good don't you ?"
        token_ids = tokenizer(prompt_tot, add_special_tokens=False)['input_ids']
        token_ids = token_ids[:max_tokens]
        return tokenizer.decode(token_ids)
    
    # Nominal case
    max_tokens = 500
    prompt = "Hey"
    body = {"prompt": prompt, "max_tokens": max_tokens}
    response = test_complete_client.post("/tests/generate", json=body)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["responses"] == [get_response(tokenizer, prompt, 0, max_tokens)]
    assert response_json["finish_reasons"] == ["stop"]
    assert set(response_json) == {"responses", "finish_reasons"}

    # Several responses and prompt_in_response
    max_tokens = 500
    prompt = "Hello there"
    body = {"prompt": prompt, "max_tokens": max_tokens, "n": 3, "prompt_in_response": True}
    response = test_complete_client.post("/tests/generate", json=body)
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json["responses"]) == 3
    assert len(response_json["finish_reasons"]) == 3
    for i in [0, 1, 2]:
        target_response = get_response(tokenizer, prompt, i, max_tokens)
        assert response_json["responses"][i] == target_response
        if target_response[-4:] == "ou ?":
            assert response_json["finish_reasons"][i] == "stop"
        else:
            assert response_json["finish_reasons"][i] == "length"
    assert response_json["prompt"] == "Hello there"
    assert set(response_json) == {"responses", "finish_reasons", "prompt"}

    # Generations stopped
    max_tokens = 5
    prompt = "Hey"
    body = {"prompt": prompt, "max_tokens": max_tokens, "n": 3}
    response = test_complete_client.post("/tests/generate", json=body)
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json["responses"]) == 3
    assert len(response_json["finish_reasons"]) == 3
    for i in [0, 1, 2]:
        target_response = get_response(tokenizer, prompt, i, max_tokens)
        assert response_json["responses"][i] == target_response
        if target_response[-4:] == "ou ?":
            assert response_json["finish_reasons"][i] == "stop"
        else:
            assert response_json["finish_reasons"][i] == "length"
    assert set(response_json) == {"responses", "finish_reasons"}

    # Some Generation stopped
    max_tokens = 18
    prompt = "Hey"
    body = {"prompt": prompt, "max_tokens": max_tokens, "n": 3}
    response = test_complete_client.post("/tests/generate", json=body)
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json["responses"]) == 3
    assert len(response_json["finish_reasons"]) == 3
    for i in [0, 1, 2]:
        target_response = get_response(tokenizer, prompt, i, max_tokens)
        assert response_json["responses"][i] == target_response
        if target_response[-4:] == "ou ?":
            assert response_json["finish_reasons"][i] == "stop"
        else:
            assert response_json["finish_reasons"][i] == "length"
    assert set(response_json) == {"responses", "finish_reasons"}


def test_tokenizer(test_complete_client: TestClient):
    """Test the functional route /tokenizer"""
    model = init_model()
    # Vanilla
    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        body = {"text": text, "vanilla": True}
        response = test_complete_client.post("/tests/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = model._tokenizer(text)['input_ids']
        assert response_json["tokens_ids"] == target_tokens_ids
        assert response_json["tokens_nb"] == len(target_tokens_ids)
        assert set(response_json) == {"tokens_ids", "tokens_nb"}

        # With with_tokens_str
        body = {"text": text, "with_tokens_str": True, "vanilla": True}
        response = test_complete_client.post("/tests/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = model._tokenizer(text)['input_ids']
        assert response_json["tokens_ids"] == target_tokens_ids
        assert response_json["tokens_nb"] == len(target_tokens_ids)
        tokens_str = model._tokenizer.convert_ids_to_tokens(target_tokens_ids)
        assert response_json["tokens_str"] == tokens_str
        assert set(response_json) == {"tokens_ids", "tokens_nb", "tokens_str"}

    #Non Vanilla
    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        body = {"text": text, "vanilla": False}
        response = test_complete_client.post("/tests/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = list(utils.proper_tokenization(model._tokenizer, text))
        assert response_json["tokens_ids"] == target_tokens_ids
        assert response_json["tokens_nb"] == len(target_tokens_ids)
        assert set(response_json) == {"tokens_ids", "tokens_nb"}

        # With with_tokens_str
        body = {"text": text, "with_tokens_str": True, "vanilla": False}
        response = test_complete_client.post("/tests/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = list(utils.proper_tokenization(model._tokenizer, text))
        assert response_json["tokens_ids"] == target_tokens_ids
        assert response_json["tokens_nb"] == len(target_tokens_ids)
        tokens_str = [utils.proper_decode(model._tokenizer, token_id) for token_id in target_tokens_ids]
        assert response_json["tokens_str"] == tokens_str
        assert set(response_json) == {"tokens_ids", "tokens_nb", "tokens_str"}


def test_decode(test_complete_client: TestClient):
    """Test the functional route /decode"""
    model = init_model()

    # Vanilla
    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        token_ids = model._tokenizer(text)['input_ids']

        body = {'token_ids': token_ids, "vanilla": True}
        response = test_complete_client.post("/tests/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["decoded_string"] == model._tokenizer.decode(token_ids)
        assert set(response_json) == {"decoded_string"}

        # With with_tokens_str
        body = {'token_ids': token_ids, "vanilla": True, "with_tokens_str": True}
        response = test_complete_client.post("/tests/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["decoded_string"] == model._tokenizer.decode(token_ids)
        assert response_json["tokens_str"] == model._tokenizer.convert_ids_to_tokens(token_ids)
        assert set(response_json) == {"decoded_string", "tokens_str"}

    #Non Vanilla
    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        token_ids = model._tokenizer(text)['input_ids']

        body = {'token_ids': token_ids, "vanilla": False}
        response = test_complete_client.post("/tests/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["decoded_string"] == utils.proper_decode(model._tokenizer, token_ids)
        assert set(response_json) == {"decoded_string"}

        # With with_tokens_str
        body = {'token_ids': token_ids, "vanilla": False, "with_tokens_str": True}
        response = test_complete_client.post("/tests/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["decoded_string"] == utils.proper_decode(model._tokenizer, token_ids)
        assert response_json["tokens_str"] == [utils.proper_decode(model._tokenizer, token_id) for token_id in token_ids]
        assert set(response_json) == {"decoded_string", "tokens_str"}


def test_split_text(test_complete_client: TestClient):
    """Test the route split_text thanks to the test_complete_client we created in conftest.py"""
    text = "Hey, my name is LLM. How are you ? Fine, you ? That's wonderful news : I'm also fine. But do you think it will last ?"

    body = {"text": text,
            "num_tokens_in_chunk": 2}
    response = test_complete_client.post("/tests/split_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    target_split_text = ["Hey, my name is LLM.",
                        " How are you ?", 
                        " Fine, you ?",
                        " That's wonderful news : I'm also fine.",
                        " But do you think it will last ?"]
    assert json_response == {"split_text": target_split_text}

    body = {"text": text,
            "num_tokens_in_chunk": 6}
    response = test_complete_client.post("/tests/split_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    target_split_text = ["Hey, my name is LLM.",
                        " How are you ? Fine, you ?",
                        " That's wonderful news : I'm also fine.",
                        " But do you think it will last ?"]
    assert json_response == {"split_text": target_split_text}

    body = {"text": text,
            "num_tokens_in_chunk": 1000}
    response = test_complete_client.post("/tests/split_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    target_split_text = [text]
    assert json_response == {"split_text": target_split_text}

    body = {"text": text,
            "num_tokens_in_chunk": 2,
            "separators": [" ?"]}
    response = test_complete_client.post("/tests/split_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    target_split_text = target_split_text = ["Hey, my name is LLM. How are you ?",
                        " Fine, you ?",
                        " That's wonderful news : I'm also fine. But do you think it will last ?"]
    assert json_response == {"split_text": target_split_text}


def test_metadata_text(test_complete_client: TestClient):
    """Test the route metadata_text thanks to the test_complete_client we created in conftest.py"""
    text = "Hey, my name is LLM. How are you ? Fine, and you ? Great."
    model = init_model()
    tokenizer = model._tokenizer

    text_tot = text * 73
    truncation_side = "left"
    max_length = 1234
    body = {"text": text_tot,
            "truncation_side": truncation_side,
            "max_length": max_length}
    response = test_complete_client.post("/tests/metadata_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    truncated_text = model.extract_text_outside_truncation(text_tot, truncation_side, max_length)
    assert json_response["tokens_nb"] == len(utils.proper_tokenization(tokenizer, text_tot))
    assert json_response["truncated_text"] == truncated_text

    text_tot = text * 73
    truncation_side = "right"
    max_length = 1234
    body = {"text": text_tot,
            "truncation_side": truncation_side,
            "max_length":1234}
    response = test_complete_client.post("/tests/metadata_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    truncated_text = model.extract_text_outside_truncation(text_tot, truncation_side, max_length)
    assert json_response["tokens_nb"] == len(utils.proper_tokenization(tokenizer, text_tot))
    assert json_response["truncated_text"] == truncated_text

    text_tot = text
    truncation_side = "left"
    max_length = 1234
    body = {"text": text,
            "truncation_side": "left",
            "max_length":1234}
    response = test_complete_client.post("/tests/metadata_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    truncated_text = model.extract_text_outside_truncation(text_tot, truncation_side, max_length)
    assert json_response["tokens_nb"] == len(utils.proper_tokenization(tokenizer, text_tot))
    assert json_response["truncated_text"] == truncated_text