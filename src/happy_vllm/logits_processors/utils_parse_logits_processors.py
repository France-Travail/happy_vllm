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

from happy_vllm.logits_processors.json_format import VLLMLogitsProcessorJSON
from happy_vllm.logits_processors.response_pool import VLLMLogitsProcessorResponsePool

# In the following parser, we have entries of the form:
# api_keyword_main : {'class': a_logits_processor_class
#                'arguments': {api_keyword: argument_of_the_class,
#                              api_keyword: argument_of_the_class}, ...},
#                  'tokenizer': False, 'model': False, 'prompt': False
# The purpose of this parser is that, if the api_keyword_main is present in the json sent via API, then
# we know we must use the corresponding logits processor. Then we look for all keyword in the 'arguments' 
# dictionary which we each corresponds to an argument of the init of the class except for the tokenizer, the model or the prompt 
# which are passed by the tokenizer, model or prompt boolean values if needed

logits_processors_parser = {'response_pool': {'class': VLLMLogitsProcessorResponsePool,
                                                'arguments': {'response_pool': 'possible_responses'},
                                                'tokenizer': True},
                            'json_format':{'class': VLLMLogitsProcessorJSON,
                                            'arguments': {'json_format': 'json_format',
                                                          'json_format_is_json_schema': 'json_format_is_json_schema'},
                                            'tokenizer_lmformatenforcer': True}}

logits_processors_incompatibilities = [('response_pool', 'json_format')]

def detect_logits_processors_incompatibilities(request_dict: dict) -> None:
    """Detects if some keywords present in request_dict could allow incompatible logits_processors
    to be present

    Args:
        request_dict (dict) : The body of the request with the different keywords
    """
    for first_keyword, second_keyword in logits_processors_incompatibilities:
        if first_keyword in request_dict and second_keyword in request_dict:
            raise ValueError(f"Two keywords are incompatible : {first_keyword} and {second_keyword}")