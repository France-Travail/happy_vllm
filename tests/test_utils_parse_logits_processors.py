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

import pytest
from happy_vllm.logits_processors.utils_parse_logits_processors import detect_logits_processors_incompatibilities


def test_detect_logits_processors_incompatibilities():
    request_dict = {"prompt": "my_prompt", "response_pool": ["Yes", "No"]}
    assert detect_logits_processors_incompatibilities(request_dict) is None

    request_dict = {"prompt": "my_prompt", "json_format": {"name": "Smith"}}
    assert detect_logits_processors_incompatibilities(request_dict) is None

    request_dict = {"prompt": "my_prompt", "response_pool": ["Yes", "No"], "json_format": {"name": "Smith"}}
    with pytest.raises(ValueError):
        detect_logits_processors_incompatibilities(request_dict)