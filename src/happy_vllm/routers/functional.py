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


import json
from fastapi import APIRouter
from vllm.utils import random_uuid
from starlette.requests import Request
from typing import AsyncGenerator, Tuple
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from lmformatenforcer import TokenEnforcerTokenizerData
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from starlette.responses import JSONResponse, Response, StreamingResponse

from happy_vllm import utils
from happy_vllm.logits_processors.min_tokens import VLLMLogitsProcessorMinTokens
from happy_vllm.logits_processors.utils_parse_logits_processors import logits_processors_parser, detect_logits_processors_incompatibilities

from ..model.model_base import Model
from ..core.resources import RESOURCE_MODEL, RESOURCES

# Functional router
router = APIRouter()


def parse_logits_processors(request_dict: dict, prompt: str, model: AsyncLLMEngine, tokenizer: PreTrainedTokenizerBase,
                            tokenizer_lmformatenforcer: TokenEnforcerTokenizerData) -> list:
    """Parses the body of the request in order to provide the logits processors
    
    Args:
        request_dict (dict): The body of the request
        model (AsyncLLMEngine): The model
        tokenizer : The tokenizer
        tokenizer_lmformatenforcer : The LM format enforcer version of the tokenizer

    Returns:
        list : The list of logits processors
    """
    logits_processors = []
    references = {'prompt': prompt, 'model': model, 'tokenizer': tokenizer, 'tokenizer_lmformatenforcer': tokenizer_lmformatenforcer}
    list_keyword = list(request_dict)
    for keyword_main in list_keyword:
        # Parse the body sent via API as explained in happy_vllm.logits_processors.utils_parse_logits_processors
        if keyword_main in logits_processors_parser:
            dict_logits_processor = logits_processors_parser[keyword_main]
            class_to_instantiate = dict_logits_processor['class']
            kwargs = {}     
            # Add prompt, model or tokenizer if needed
            for reference, refered_object in references.items():
                if dict_logits_processor.get(reference, False):
                    kwargs[reference] = refered_object
            # Add arguments to instantiate the logits_processor if needed
            for keyword, class_argument in dict_logits_processor['arguments'].items():
                if keyword in request_dict:
                    kwargs[class_argument] = request_dict.pop(keyword)
            logit_processor = class_to_instantiate(**kwargs)
            logits_processors.append(logit_processor)
    # To ensure that no unsuitable keywords are passed to vllm, we pop them
    for keyword_main, logits_processor_config in logits_processors_parser.items():
        for keyword in logits_processor_config['arguments']:
            if keyword in request_dict:
                request_dict.pop(keyword)
    return logits_processors


def parse_generate_parameters(request_dict: dict, model: AsyncLLMEngine, tokenizer: PreTrainedTokenizerBase,
                            tokenizer_lmformatenforcer: TokenEnforcerTokenizerData) -> Tuple[str, bool, SamplingParams]:
    """Parses the body of the request to obtain the prompt and the sampling parameters

    Args:
        request_dict (dict): The body of the request
        model (AsyncLLMEngine): The model
        tokenizer : The tokenizer
        tokenizer_lmformatenforcer : The LM format enforcer version of the tokenizer
    Returns:
        str : The prompt
        bool : Whether the prompt should be displayed in the response
        SamplingParams : The vllm sampling parameters
    """
    prompt = request_dict.pop("prompt")
    if 'prompt_in_response' in request_dict:
        prompt_in_response = request_dict.pop('prompt_in_response')
    else:
        prompt_in_response = False
    detect_logits_processors_incompatibilities(request_dict)
    logits_processors = parse_logits_processors(request_dict, prompt, model, tokenizer, tokenizer_lmformatenforcer)
    sampling_params = SamplingParams(**request_dict)
    sampling_params.logits_processors = logits_processors
    for logits_processor in logits_processors:
        if isinstance(logits_processor, VLLMLogitsProcessorMinTokens):
            min_tokens = logits_processor.min_tokens
            if min_tokens > sampling_params.max_tokens:
                raise ValueError(f"min_tokens : {min_tokens} can't be superior to max_tokens : {sampling_params.max_tokens}")
    return prompt, prompt_in_response, sampling_params


@router.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: The prompt to use for the generation.
    - other fields: The sampling parameters (See `SamplingParams` for details).
    """
    
    model: Model = RESOURCES.get(RESOURCE_MODEL)
    request_dict = await request.json()
    prompt, prompt_in_response, sampling_params = parse_generate_parameters(request_dict, model._model, model._tokenizer, model._tokenizer_lmformatenforcer)
    request_id = random_uuid()
    model._tokenizer.truncation_side = model.original_truncation_side
    results_generator = model._model.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await model._model.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    if final_output is None:
        raise ValueError('The final ouput is None')
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    finish_reasons = [output.finish_reason for output in request_output.outputs]
    finish_reasons = ["None" if finish_reason is None else finish_reason for finish_reason in finish_reasons]
    ret = {"responses": text_outputs, "finish_reasons": finish_reasons}
    if prompt_in_response:
        ret['prompt'] = prompt
    return JSONResponse(ret)


@router.post("/generate_stream")
async def generate_stream(request: Request) -> StreamingResponse:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: The prompt to use for the generation.
    - other fields: The sampling parameters (See `SamplingParams` for details).
    """
    model: Model = RESOURCES.get(RESOURCE_MODEL)
    request_dict = await request.json()
    prompt, prompt_in_response, sampling_params = parse_generate_parameters(request_dict, model._model, model._tokenizer, model._tokenizer_lmformatenforcer)
    request_id = random_uuid()
    model._tokenizer.truncation_side = model.original_truncation_side
    results_generator = model._model.generate(prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[str, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                output.text for output in request_output.outputs
            ]
            finish_reasons = [output.finish_reason for output in request_output.outputs]
            finish_reasons = ["None" if finish_reason is None else finish_reason for finish_reason in finish_reasons]
            ret = {"responses": text_outputs, "finish_reasons": finish_reasons}
            if prompt_in_response:
                ret['prompt'] = prompt
            yield (json.dumps(ret) + "\n")#.encode("utf-8")

    return StreamingResponse(stream_results())


@router.post("/tokenizer")
async def tokenizer(request: Request) -> Response:
    """Tokenizes a text

    The request should be a JSON object with the following fields:
    - text: The text to tokenize
    - with_tokens_str (optional): Whether we want the tokens strings in the output
    - vanilla (optional) : Whether we want the vanilla version of the tokenizers
    """
    model: Model = RESOURCES.get(RESOURCE_MODEL)
    request_dict = await request.json()
    text = request_dict.pop("text")
    vanilla = request_dict.get("vanilla", True)
    with_tokens_str = request_dict.get('with_tokens_str', False)
    
    if vanilla:
        tokens_ids = model._tokenizer(text)['input_ids']
        if with_tokens_str:
            tokens_str = model._tokenizer.convert_ids_to_tokens(tokens_ids)
    else:
        tokens_ids = model.tokenize(text)
        if with_tokens_str:
            tokens_str = [utils.proper_decode(model._tokenizer, token_id) for token_id in tokens_ids]
            

    ret = {"tokens_ids": tokens_ids, "tokens_nb": len(tokens_ids)}
    if with_tokens_str:
        ret['tokens_str'] = tokens_str
    return JSONResponse(ret)


@router.post("/decode")
async def decode(request: Request) -> Response:
    """Decodes token ids

    The request should be a JSON object with the following fields:
    - token_ids: The ids of the tokens
    - with_tokens_str : If the result should also include a list of str
    - vanilla (optional) : Whether we want the vanilla version of the tokenizers
    """
    model: Model = RESOURCES.get(RESOURCE_MODEL)
    request_dict = await request.json()
    token_ids = request_dict.pop("token_ids")
    with_tokens_str = request_dict.get("with_tokens_str", False)
    vanilla = request_dict.get("vanilla", True)
    
    if vanilla:
        decoded_string = model._tokenizer.decode(token_ids)
        if with_tokens_str:
            tokens_str = model._tokenizer.convert_ids_to_tokens(token_ids)
    else:
        decoded_string = utils.proper_decode(model._tokenizer, token_ids)
        if with_tokens_str:
            tokens_str = [utils.proper_decode(model._tokenizer, token_id) for token_id in token_ids]
            

    ret = {"decoded_string": decoded_string}
    if with_tokens_str:
        ret[ "tokens_str"] = tokens_str
    return JSONResponse(ret)


@router.post("/split_text")
async def split_text(request: Request):
    """Splits a text

    The request should be a JSON object with the following fields:
    - text: The text to split
    - num_tokens_in_chunk (optional): The minimal number of tokens we want in each split
    - separators (optional) : The allowed separators between the chunks
    """
    model: Model = RESOURCES.get(RESOURCE_MODEL)

    request_dict = await request.json()
    split_text = model.split_text(**request_dict)
    response = {"split_text": split_text}

    return JSONResponse(response)


@router.post("/metadata_text")
async def metadata_text(request: Request):
    """Gives meta data on a text

    The request should be a JSON object with the following fields:
    - text: The text to parse
    - truncation_side (optional): The truncation side of the tokenizer
    - max_length (optional) : The max length before truncation

    The default values for truncation_side and max_length are those of the underlying model
    """
    model: Model = RESOURCES.get(RESOURCE_MODEL)

    request_dict = await request.json()

    tokens_ids = model.tokenize(request_dict['text'])
    truncated_text = model.extract_text_outside_truncation(**request_dict)
    ret = {"tokens_nb": len(tokens_ids), "truncated_text": truncated_text}

    return JSONResponse(ret)