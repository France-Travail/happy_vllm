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

import torch
from typing import List, Union
from lmformatenforcer.integrations.vllm import VLLMLogitsProcessor
from lmformatenforcer import TokenEnforcerTokenizerData, JsonSchemaParser, TokenEnforcer


def check_simple_json(json_format: dict) -> bool:
    """Checks if the simple_json given is of the right form : a flat dictionary where the values are in the list ['string', 'integer', 'boolean', 'number']
    or is a list with a single element in it belonging to the list ['string', 'integer', 'boolean', 'number'] and the keywords are strings.
    So a dictionary of the following type:
    {"first_keyword": "string",
    "second_keyword": "integer",
    "third_keyword": "boolean",
    "fourth_keyword": "number",
    "fifth_keyword": ["string"]}

    Args:
        json_format (dict) : The dictionary containing the key and value we want to check
    """
    for key, value in json_format.items():
        if not isinstance(key, str):
            raise ValueError(f'The keyword {key} in the json schema is not a string')
        possible_values = ['string', 'integer', 'boolean', 'number']
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f'The value {value} corresponding to the keyword {key} appears to be a list not composed of a single element in {possible_values}')
            if value[0] not in possible_values:
                raise ValueError(f'The value {value} corresponding to the keyword {key} is a list where the single element in not in {possible_values}')
        elif value not in possible_values:
            raise ValueError(f'The value {value} corresponding to the keyword {key} is not in {possible_values}')
    return True


def get_json_format_parser(json_format: dict, is_json_schema: bool) -> JsonSchemaParser:
    """Get the JsonSchemaParser from the json format indicated. This json format can be of a simple form as described in check_simple_json
    or is a valid json schema as described in https://json-schema.org/learn/getting-started-step-by-step

    Args:
        json_format (dict) : The json format we want
        is_json_schema (bool) : Indicates if the json_format is a valid json schema or not

    Returns:
        The lmformatenforcer JsonSchemaParser

    """
    if is_json_schema:
        schema = json_format
    else:
        # Check if json_format is indeed a json in a simple form
        check_simple_json(json_format)
        schema = {'properties':{},
                'required':[],
                'title':'simple_json',
                'type': 'object'}
        for key, type_value in json_format.items():
            if isinstance(type_value, list):
                schema['properties'][key] = {'type': 'array', 'items': {"type": type_value[0]}}
            else:
                schema['properties'][key] = {'type': type_value}
            schema['required'].append(key)

    json_format_parser = JsonSchemaParser(schema)
    return json_format_parser


class VLLMLogitsProcessorJSON(VLLMLogitsProcessor):

    def __init__(self, tokenizer_lmformatenforcer: TokenEnforcerTokenizerData, json_format: Union[dict, None], json_format_is_json_schema: bool = False) -> None:
        '''Initializes the logits processor for json outputs

        Args:
            tokenizer_lmformatenforcer : The tokenizer obtained via lm-format-enforcer
            json_format : The json format we want

        Kwargs:
            json_format_is_json_schema (bool) : Indicates if the json_format is a valid json schema or not
        '''
        # Tests if the input is relevant and thus if this logits processor should do nothing
        if json_format == {} or json_format is None:
            self.logits_processor_activated = False
        else:
            self.logits_processor_activated = True
            json_format_parser = get_json_format_parser(json_format, json_format_is_json_schema)
            token_enforcer = TokenEnforcer(tokenizer_lmformatenforcer, json_format_parser)
            super().__init__(token_enforcer, False)
    
    def __call__(self, input_ids: List[int], scores: torch.Tensor):
        if self.logits_processor_activated :
            return super(VLLMLogitsProcessorJSON, self).__call__(input_ids, scores)
        else:
            return scores
            