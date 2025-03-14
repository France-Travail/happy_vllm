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

from httpx import AsyncClient
from fastapi import HTTPException
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.openai import protocol as vllm_protocol

from happy_vllm import utils
from .conftest import TEST_MODELS_DIR
from happy_vllm.routers import functional
from happy_vllm.model.model_base import Model


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


def test_verify_request():
    # Without HTTPException
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        temperature=1.0,
        echo=True,
        stream=None
    )
    assert functional.verify_request(request) == None

    # With HTTPException echo and stream
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        echo=True,
        stream=True
    )
    with pytest.raises(HTTPException) as error:
        functional.verify_request(request)
    assert error.value.detail == "Use both echo and stream breaks backend"

    # With HTTPException temperature and top_p equals to 0
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        temperature=0,
        top_p=0
    )
    with pytest.raises(HTTPException) as error:
        functional.verify_request(request)
    assert error.value.detail == "Use temperature and top_p equal to 0 breaks the model"

    # With HTTPException high temperature and top_k equals to 1
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        temperature=2.4,
        top_k=1
    )
    with pytest.raises(HTTPException) as error:
        functional.verify_request(request)
    assert error.value.detail == "Use temperature with high value: 2.4 and top_k equals to 1 : 1 breaks the model"

    # With HTTPException top_p and top_k equals to 1
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        top_p=1,
        top_k=1
    )
    with pytest.raises(HTTPException) as error:
        functional.verify_request(request)
    assert error.value.detail == "Use top_p and top_k equal to 1 breaks the model"

    # With HTTPException max_tokens less than min_tokens
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        max_tokens=50,
        min_tokens=100
    )
    with pytest.raises(HTTPException) as error:
        functional.verify_request(request)
    assert error.value.detail == "Use max_tokens: 50 less than min_tokens : 100 breaks the model"


@pytest.mark.asyncio
async def test_tokenizer(test_complete_client: AsyncClient):
    """Test the functional route /v1/tokenizer"""
    model = init_model()
    # Vanilla
    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        body = {"text": text, "vanilla": True}
        response = await test_complete_client.post("/tests/v1/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = model._tokenizer(text)['input_ids']
        assert response_json["tokens_ids"] == target_tokens_ids
        assert response_json["tokens_nb"] == len(target_tokens_ids)
        assert set(response_json) == {"tokens_ids", "tokens_nb"}

        # With with_tokens_str
        body = {"text": text, "with_tokens_str": True, "vanilla": True}
        response = await test_complete_client.post("/tests/v1/tokenizer", json=body)
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
        response = await test_complete_client.post("/tests/v1/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = list(utils.proper_tokenization(model._tokenizer, text))
        assert response_json["tokens_ids"] == target_tokens_ids
        assert response_json["tokens_nb"] == len(target_tokens_ids)
        assert set(response_json) == {"tokens_ids", "tokens_nb"}

        # With with_tokens_str
        body = {"text": text, "with_tokens_str": True, "vanilla": False}
        response = await test_complete_client.post("/tests/v1/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = list(utils.proper_tokenization(model._tokenizer, text))
        assert response_json["tokens_ids"] == target_tokens_ids
        assert response_json["tokens_nb"] == len(target_tokens_ids)
        tokens_str = [utils.proper_decode(model._tokenizer, token_id) for token_id in target_tokens_ids]
        assert response_json["tokens_str"] == tokens_str
        assert set(response_json) == {"tokens_ids", "tokens_nb", "tokens_str"}


@pytest.mark.asyncio
async def test_tokenizer_v2(test_complete_client: AsyncClient):
    """Test the functional route /v2/tokenizer"""
    model = init_model()
    # completions
    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        # With add_special_tokens
        body = {"model": "test", "prompt": text, "add_special_tokens": True}
        response = await test_complete_client.post("/tests/v2/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = model._tokenizer(body["prompt"], add_special_tokens=body['add_special_tokens'])['input_ids']
        assert response_json["tokens"] == target_tokens_ids
        assert response_json["count"] == len(target_tokens_ids)
        assert set(response_json) == {"count", "max_model_len", "tokens"}

        # Without add_special_tokens
        body = {"model": "test", "prompt": text, "add_special_tokens": False}
        response = await test_complete_client.post("/tests/v2/tokenizer", json=body)
        assert response.status_code == 200
        response_json = response.json()
        target_tokens_ids = model._tokenizer(body["prompt"], add_special_tokens=body['add_special_tokens'])['input_ids']
        assert response_json["tokens"] == target_tokens_ids
        assert response_json["count"] == len(target_tokens_ids)
        assert set(response_json) == {"count", "max_model_len", "tokens"}


@pytest.mark.asyncio
async def test_decode(test_complete_client: AsyncClient):
    """Test the functional route /v1/decode"""
    model = init_model()

    # Vanilla
    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        token_ids = model._tokenizer(text)['input_ids']

        body = {'token_ids': token_ids, "vanilla": True}
        response = await test_complete_client.post("/tests/v1/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["decoded_string"] == model._tokenizer.decode(token_ids)
        assert set(response_json) == {"decoded_string"}

        # With with_tokens_str
        body = {'token_ids': token_ids, "vanilla": True, "with_tokens_str": True}
        response = await test_complete_client.post("/tests/v1/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["decoded_string"] == model._tokenizer.decode(token_ids)
        assert response_json["tokens_str"] == model._tokenizer.convert_ids_to_tokens(token_ids)
        assert set(response_json) == {"decoded_string", "tokens_str"}

    #Non Vanilla
    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        token_ids = model._tokenizer(text)['input_ids']

        body = {'token_ids': token_ids, "vanilla": False}
        response = await test_complete_client.post("/tests/v1/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["decoded_string"] == utils.proper_decode(model._tokenizer, token_ids)
        assert set(response_json) == {"decoded_string"}

        # With with_tokens_str
        body = {'token_ids': token_ids, "vanilla": False, "with_tokens_str": True}
        response = await test_complete_client.post("/tests/v1/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["decoded_string"] == utils.proper_decode(model._tokenizer, token_ids)
        assert response_json["tokens_str"] == [utils.proper_decode(model._tokenizer, token_id) for token_id in token_ids]
        assert set(response_json) == {"decoded_string", "tokens_str"}


@pytest.mark.asyncio
async def test_decode_v2(test_complete_client: AsyncClient):
    """Test the functional route /v2/decode"""
    model = init_model()

    for text in ["How do you do?", "I am Lliam. How are you ?", "Marvelous, it works !"]:
        token_ids = model._tokenizer(text)['input_ids']

        body = {'tokens': token_ids, "model": "test"}
        response = await test_complete_client.post("/tests/v2/decode", json=body)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["prompt"] == model._tokenizer.decode(token_ids)
        assert set(response_json) == {"prompt"}


@pytest.mark.asyncio
async def test_split_text(test_complete_client: AsyncClient):
    """Test the route /v1/split_text thanks to the test_complete_client we created in conftest.py"""
    text = "Hey, my name is LLM. How are you ? Fine, you ? That's wonderful news : I'm also fine. But do you think it will last ?"

    body = {"text": text,
            "num_tokens_in_chunk": 2}
    response = await test_complete_client.post("/tests/v1/split_text", json=body)
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
    response = await test_complete_client.post("/tests/v1/split_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    target_split_text = ["Hey, my name is LLM.",
                        " How are you ? Fine, you ?",
                        " That's wonderful news : I'm also fine.",
                        " But do you think it will last ?"]
    assert json_response == {"split_text": target_split_text}

    body = {"text": text,
            "num_tokens_in_chunk": 1000}
    response = await test_complete_client.post("/tests/v1/split_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    target_split_text = [text]
    assert json_response == {"split_text": target_split_text}

    body = {"text": text,
            "num_tokens_in_chunk": 2,
            "separators": [" ?"]}
    response = await test_complete_client.post("/tests/v1/split_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    target_split_text = target_split_text = ["Hey, my name is LLM. How are you ?",
                        " Fine, you ?",
                        " That's wonderful news : I'm also fine. But do you think it will last ?"]
    assert json_response == {"split_text": target_split_text}


@pytest.mark.asyncio
async def test_metadata_text(test_complete_client: AsyncClient):
    """Test the route /v1/metadata_text thanks to the test_complete_client we created in conftest.py"""
    text = "Hey, my name is LLM. How are you ? Fine, and you ? Great."
    model = init_model()
    tokenizer = model._tokenizer

    text_tot = text * 73
    truncation_side = "left"
    max_length = 1234
    body = {"text": text_tot,
            "truncation_side": truncation_side,
            "max_length": max_length}
    response = await test_complete_client.post("/tests/v1/metadata_text", json=body)
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
    response = await test_complete_client.post("/tests/v1/metadata_text", json=body)
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
    response = await test_complete_client.post("/tests/v1/metadata_text", json=body)
    assert response.status_code == 200
    json_response = response.json()
    truncated_text = model.extract_text_outside_truncation(text_tot, truncation_side, max_length)
    assert json_response["tokens_nb"] == len(utils.proper_tokenization(tokenizer, text_tot))
    assert json_response["truncated_text"] == truncated_text