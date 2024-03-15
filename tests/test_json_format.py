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
from transformers import AutoTokenizer
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

from happy_vllm import utils
from happy_vllm.logits_processors import json_format as json_format_module
from happy_vllm.logits_processors.json_format import VLLMLogitsProcessorJSON

from .conftest import TEST_MODELS_DIR


def teardown_module():
    if os.path.exists(TEST_MODELS_DIR):
        if os.path.isdir(TEST_MODELS_DIR):
            shutil.rmtree(TEST_MODELS_DIR)


def test_check_simple_json():
    # Nominal case
    simple_json = {'first_keyword': 'string',
                    'second_keyword': 'integer',
                    'third_keyword': 'boolean',
                    'fourth_keyword': 'number',
                    'fifth_keyword': ['string'],
                    'sixth_keyword': ['integer'],
                    'seventh_keyword': ['boolean'],
                    'eighth_keyword': ['number']}
    assert json_format_module.check_simple_json(simple_json)
    
    # One wrong value
    simple_json = {'first_keyword': 'string',
                    'second_keyword': 'integer',
                    'third_keyword': 'boolean',
                    'fourth_keyword': 'float',
                    'fifth_keyword': ['string'],
                    'sixth_keyword': ['integer'],
                    'seventh_keyword': ['boolean'],
                    'eighth_keyword': ['number']}
    with pytest.raises(ValueError) as error:
        json_format_module.check_simple_json(simple_json)

    # One wrong value in a list
    simple_json = {'first_keyword': 'string',
                    'second_keyword': 'integer',
                    'third_keyword': 'boolean',
                    'fourth_keyword': 'number',
                    'fifth_keyword': ['string'],
                    'sixth_keyword': ['integer'],
                    'seventh_keyword': ['boolean'],
                    'eighth_keyword': ['float']}
    with pytest.raises(ValueError) as error:
        json_format_module.check_simple_json(simple_json)
    
    # List with zero element
    simple_json = {'first_keyword': 'string',
                    'second_keyword': 'integer',
                    'third_keyword': 'boolean',
                    'fourth_keyword': 'number',
                    'fifth_keyword': ['string'],
                    'sixth_keyword': ['integer'],
                    'seventh_keyword': ['boolean'],
                    'eighth_keyword': []}
    with pytest.raises(ValueError) as error:
        json_format_module.check_simple_json(simple_json)

    # List with more than one element
    simple_json = {'first_keyword': 'string',
                    'second_keyword': 'integer',
                    'third_keyword': 'boolean',
                    'fourth_keyword': 'number',
                    'fifth_keyword': ['string'],
                    'sixth_keyword': ['integer'],
                    'seventh_keyword': ['boolean'],
                    'eighth_keyword': ['string', 'integer']}
    with pytest.raises(ValueError) as error:
        json_format_module.check_simple_json(simple_json)

    # One wrong keyword
    simple_json = {1: 'string',
                    'second_keyword': 'integer',
                    'third_keyword': 'boolean',
                    'fourth_keyword': 'number'}
    with pytest.raises(ValueError) as error:
        json_format_module.check_simple_json(simple_json)


def test_get_json_format_parser():
    simple_json = {'first_keyword': 'string',
                    'second_keyword': 'integer',
                    'third_keyword': 'boolean',
                    'fourth_keyword': 'number',
                    'fifth_keyword': ['string'],
                    'sixth_keyword': ['integer'],
                    'seventh_keyword': ['boolean'],
                    'eighth_keyword': ['number']}
    json_schema = {"properties": 
               {"name": {"type": "string"},
                "surname": {"type": "string"},
                "favorite_food": {"type": "string"},
                "current_number_of_children": {"type": "integer"},
                "work": {"type": "object", "properties": {"team_name": {"type": "string"}, "job_title": {"type": "string"}}, "required":["team_name", "job_title"]}},
                "required": ["name", "surname", "favorite_food", "current_number_of_children", "work"], "title": "json_format", "type": "object"}
    
    # Nominal case, simple json
    json_format = simple_json
    json_format_parser = json_format_module.get_json_format_parser(json_format=json_format, is_json_schema=False)
    assert isinstance(json_format_parser, JsonSchemaParser)

    # Nominal case, json schema
    json_format = json_schema
    json_format_parser = json_format_module.get_json_format_parser(json_format=json_format, is_json_schema=True)
    assert isinstance(json_format_parser, JsonSchemaParser)

    # Wrong is_json_schema for a simple json
    json_format = simple_json
    with pytest.raises(Exception) as error:
        json_format_parser = json_format_module.get_json_format_parser(json_format=json_format, is_json_schema=True)

    # Wrong is_json_schema for a json schema
    json_format = json_schema
    with pytest.raises(ValueError) as error:
        json_format_parser = json_format_module.get_json_format_parser(json_format=json_format, is_json_schema=False)


def test_VLLMLogitsProcessorJSON_init():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)
    tokenizer_lmformatenforcer = build_token_enforcer_tokenizer_data(tokenizer)

    simple_json = {'first_keyword': 'string',
                    'second_keyword': 'integer',
                    'third_keyword': 'boolean',
                    'fourth_keyword': 'number',
                    'fifth_keyword': ['string'],
                    'sixth_keyword': ['integer'],
                    'seventh_keyword': ['boolean'],
                    'eighth_keyword': ['number']}
    json_schema = {"properties": 
               {"name": {"type": "string"},
                "surname": {"type": "string"},
                "favorite_food": {"type": "string"},
                "current_number_of_children": {"type": "integer"},
                "work": {"type": "object", "properties": {"team_name": {"type": "string"}, "job_title": {"type": "string"}}, "required":["team_name", "job_title"]}},
                "required": ["name", "surname", "favorite_food", "current_number_of_children", "work"], "title": "json_format", "type": "object"}

    # Nominal case simple json
    json_format = simple_json
    json_format_is_json_schema = False
    json_logits_processor = VLLMLogitsProcessorJSON(tokenizer_lmformatenforcer, json_format, json_format_is_json_schema)
    assert isinstance(json_logits_processor, VLLMLogitsProcessorJSON)
    assert json_logits_processor.logits_processor_activated

    # Nominal case json schema
    json_format = json_schema
    json_format_is_json_schema = True
    json_logits_processor = VLLMLogitsProcessorJSON(tokenizer_lmformatenforcer, json_format, json_format_is_json_schema)
    assert isinstance(json_logits_processor, VLLMLogitsProcessorJSON)
    assert json_logits_processor.logits_processor_activated

    # Nominal case empty json format
    json_format = {}
    json_format_is_json_schema = True
    json_logits_processor = VLLMLogitsProcessorJSON(tokenizer_lmformatenforcer, json_format, json_format_is_json_schema)
    assert isinstance(json_logits_processor, VLLMLogitsProcessorJSON)
    assert not(json_logits_processor.logits_processor_activated)

    # Nominal case json format is None
    json_format = None
    json_format_is_json_schema = False
    json_logits_processor = VLLMLogitsProcessorJSON(tokenizer_lmformatenforcer, json_format, json_format_is_json_schema)
    assert isinstance(json_logits_processor, VLLMLogitsProcessorJSON)
    assert not(json_logits_processor.logits_processor_activated)


def test_VLLMLogitsProcessorJSON_call():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["tokenizer_name"],
                                                     cache_dir=TEST_MODELS_DIR)
    tokenizer_lmformatenforcer = build_token_enforcer_tokenizer_data(tokenizer)

    # Logits processor is deactivated
    json_format = None
    json_logits_processor = VLLMLogitsProcessorJSON(tokenizer_lmformatenforcer, json_format)
    scores = torch.rand((1000, ))
    assert max(scores) == pytest.approx(max(json_logits_processor([], scores)))
    assert min(scores) == pytest.approx(min(json_logits_processor([], scores)))