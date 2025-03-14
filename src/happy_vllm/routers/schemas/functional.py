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


"""Functional schemas"""

import os
import json
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field, conint
from typing import Any, List, Union, Optional
from vllm.entrypoints.openai.protocol import ResponseFormat, CompletionResponse, ChatCompletionResponse

from .utils import NumpyArrayEncoder

# Load the response examples
directory = os.path.dirname(os.path.abspath(__file__))
response_examples_path = os.path.join(directory, "examples", "response.json")
with open(response_examples_path, "r") as file:
    response_examples = json.load(file)


class NumpyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(content, cls=NumpyArrayEncoder).encode()


class ResponseTokenizer(BaseModel):
    tokens_ids: List[int] = Field(None, title="List of token ids")
    tokens_nb: int = Field(None, title="Number of tokens")
    tokens_str: List[str] = Field(None, title="List of decoded tokens")
    model_config = {"json_schema_extra": {"examples": [response_examples["tokenizer"]]}}


class RequestTokenizer(BaseModel):
    text: str = Field(None, title="Text to tokenize")
    with_token_str: bool = Field(None, title="Add strings of tokens")
    vanilla: bool = Field(None, title="Use vanilla tokenizer")


class ResponseDecode(BaseModel):
    decoded_string: str = Field(None, title="Decoded text")
    tokens_str: List[str] = Field(None, title="Decoded tokens")
    model_config = {"json_schema_extra": {"examples": [response_examples["decode"]]}}


class RequestDecode(BaseModel):
    token_ids: List[int] = Field(None, title="List of token ids")
    with_token_str: bool = Field(None, title="Add strings of tokens")
    vanilla: bool = Field(None, title="Use vanilla model")


class ResponseSplitText(BaseModel):
    split_text: List[str] = Field(None, title="Split text")
    model_config = {
        "json_schema_extra": {"examples": [response_examples["split_text"]]}
    }


class RequestSplitText(BaseModel):
    text: str = Field(None, title="Input text")
    num_tokens_in_chunk: int = Field(None, title="Min number of tokens per chunk")
    separators: List[str] = Field(None, title="List of separators")


class ResponseMetadata(BaseModel):
    tokens_nb: int = Field(None, title="Number of tokens")
    truncated_text: str = Field(None, title="Truncated text")
    model_config = {
        "json_schema_extra": {"examples": [response_examples["metadata_text"]]}
    }


class RequestMetadata(BaseModel):
    text: str = Field(None, title="Input text")
    truncation_side: str = Field(None, title="Side of truncation")
    max_length: int = Field(None, title="Max length before truncation")


class HappyvllmCompletionResponse(CompletionResponse):
    model_config = {"json_schema_extra": {"examples": [response_examples["completion_response"]]}}


class HappyvllmChatCompletionResponse(ChatCompletionResponse):
    model_config = {"json_schema_extra": {"examples": [response_examples["chat_completion_response"]]}} 


class RequestAbortRequest(BaseModel):
    request_id: str