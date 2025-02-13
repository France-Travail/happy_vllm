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
import json
import vllm.envs as envs
from vllm.utils import random_uuid
from pydantic import BaseModel, Field
from starlette.requests import Request
from typing_extensions import assert_never
from vllm.sampling_params import SamplingParams
from fastapi import APIRouter, Body, HTTPException
from vllm.entrypoints.utils import with_cancellation
from vllm.engine.async_llm_engine import AsyncLLMEngine
from lmformatenforcer import TokenEnforcerTokenizerData
from vllm.entrypoints.openai import protocol as vllm_protocol
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from starlette.responses import JSONResponse, Response, StreamingResponse
from typing import Annotated, AsyncGenerator, Tuple, List, Optional, Union
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization

from happy_vllm import utils
from happy_vllm.logits_processors.response_pool import VLLMLogitsProcessorResponsePool
from happy_vllm.logits_processors.utils_parse_logits_processors import logits_processors_parser, detect_logits_processors_incompatibilities

from ..model.model_base import Model
from ..core.resources import RESOURCE_MODEL, RESOURCES
from happy_vllm.routers.schemas import functional as functional_schema



# Load the response examples
directory = os.path.dirname(os.path.abspath(__file__))
request_examples_path = os.path.join(directory, "schemas", "examples", "request.json")
with open(request_examples_path, 'r') as file:
    request_openapi_examples = json.load(file)

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
    multi_modal_data = request_dict
    detect_logits_processors_incompatibilities(request_dict)
    logits_processors = parse_logits_processors(request_dict, prompt, model, tokenizer, tokenizer_lmformatenforcer)
    sampling_params = SamplingParams(**request_dict)
    sampling_params.logits_processors = logits_processors
    for logits_processor in logits_processors:
        if isinstance(logits_processor, VLLMLogitsProcessorResponsePool):
            min_tokens = sampling_params.min_tokens
            reponse_pool = logits_processor.possible_tokens_responses
            if min_tokens > 0 and len(reponse_pool):
                raise ValueError(f"min_tokens : {min_tokens} is incompatible with the `response_pool` keyword")
    return prompt, prompt_in_response, sampling_params


def verify_request(request: Request) -> None:
    status_code = 422
    detail = None
    if request.echo and request.stream:
        detail="Use both echo and stream breaks backend"
    if request.temperature is not None and request.top_p is not None:
        if request.temperature == 0 and request.top_p == 0:
            detail=f"Use temperature and top_p equal to 0 breaks the model"
    if request.temperature and request.top_k:
        if request.temperature > 2 and request.top_k == 1:
            detail=f"Use temperature with high value: {request.temperature} and top_k equals to 1 : {request.top_k} breaks the model"
    if request.top_p and request.top_k:
        if request.top_p == 1 and request.top_k == 1:
            detail=f"Use top_p and top_k equal to 1 breaks the model"
    if request.max_tokens and request.min_tokens:
        if request.max_tokens <= request.min_tokens:
            detail=f"Use max_tokens: {request.max_tokens} less than min_tokens : {request.min_tokens} breaks the model"
    if request.echo and (request.top_k is not None or request.top_p is not None or request.min_p is not None):
        detail=f"Use echo with top_k or top_p or min_p breaks backend"
    if request.prompt_logprobs and (request.top_k is not None or request.top_p is not None or request.min_p is not None):
        detail=f"Use prompt_logprobs with top_k or top_p or min_p breaks backend"
    if detail :
        raise HTTPException(
            status_code=status_code, 
            detail=detail
        )


def base(request: Request, model: Model) -> OpenAIServing:
    return model.openai_serving_tokenization


@router.post("/v1/generate", response_model=functional_schema.ResponseGenerate)
async def generate(
    request: Request,
    request_type: Annotated[
        functional_schema.RequestGenerate,
        Body(openapi_examples=request_openapi_examples["generate"])] = None
    ) -> Response:
    """DEPRECATED. Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: The prompt to use for the generation.
    - other fields: The sampling parameters (See `SamplingParams` from vLLM for more details : 
                                             https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).
    """
    
    model: Model = RESOURCES[RESOURCE_MODEL]
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
    prompt_nb_tokens = len(final_output.prompt_token_ids)
    nb_tokens_outputs = [len(output.token_ids) for output in final_output.outputs]
    usages = [{"prompt_tokens": prompt_nb_tokens,
               "completion_tokens": completion_tokens,
               "total_tokens": prompt_nb_tokens + completion_tokens } for completion_tokens in nb_tokens_outputs]
    finish_reasons = [output.finish_reason for output in request_output.outputs]
    finish_reasons = ["None" if finish_reason is None else finish_reason for finish_reason in finish_reasons]
    ret = {"responses": text_outputs, "finish_reasons": finish_reasons, "usages": usages}
    if prompt_in_response:
        ret['prompt'] = prompt
    return JSONResponse(ret)


@router.post("/v1/generate_stream", response_model=functional_schema.ResponseGenerate)
async def generate_stream(request: Request,
    request_type: Annotated[
        functional_schema.RequestGenerate,
        Body(openapi_examples=request_openapi_examples["generate_stream"])] = None
    ) -> StreamingResponse:
    """DEPRECATED. Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: The prompt to use for the generation.
    - other fields: The sampling parameters (See `SamplingParams` for details).
    """
    model: Model = RESOURCES[RESOURCE_MODEL]
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
            prompt_nb_tokens = len(request_output.prompt_token_ids)
            nb_tokens_outputs = [len(output.token_ids) for output in request_output.outputs]
            usages = [{"prompt_tokens": prompt_nb_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_nb_tokens + completion_tokens } for completion_tokens in nb_tokens_outputs]
            ret = {"responses": text_outputs, "finish_reasons": finish_reasons, "usages": usages}
            if prompt_in_response:
                ret['prompt'] = prompt
            yield (json.dumps(ret) + "\n")#.encode("utf-8")

    return StreamingResponse(stream_results())


@router.post("/v1/tokenizer", response_model=functional_schema.ResponseTokenizer)
async def tokenizer(request: Request,
    request_type: Annotated[
        functional_schema.RequestTokenizer,
        Body(openapi_examples=request_openapi_examples["tokenizer"])] = None
    ) -> Response:
    """Tokenizes a text

    The request should be a JSON object with the following fields:
    - text: The text to tokenize
    - with_tokens_str (optional): Whether we want the tokens strings in the output
    - vanilla (optional) : Whether we want the vanilla version of the tokenizers
    """
    model: Model = RESOURCES[RESOURCE_MODEL]
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


@router.post("/v2/tokenizer", response_model=vllm_protocol.TokenizeResponse)
@with_cancellation
async def tokenizer_v2(request: Annotated[vllm_protocol.TokenizeRequest,
        Body(openapi_examples=request_openapi_examples["vllm_tokenizer"])],
        raw_request: Request
    ):
    """Tokenizes a text

    The request should be a JSON object with the following fields:

    Completions :
    - model : ID of the model to use
    - prompt : The text to tokenize
    - add_special_tokens : Add a special tokens to the begin (optional, default value : `true`)
    
    Chat Completions:
    - model : ID of the model to use
    - messages: The texts to tokenize
    - add_special_tokens : Add a special tokens to the begin (optional, default value : `false`)
    - add_generation_prompt : Add generation prompt's model in decode response (optional, default value : `true`)
    """
    model: Model = RESOURCES[RESOURCE_MODEL]
    generator = await model.openai_serving_tokenization.create_tokenize(request, raw_request)
    if isinstance(generator, vllm_protocol.ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        if not isinstance(generator, vllm_protocol.TokenizeResponse):
            raise TypeError("Expected generator to be an instance of vllm_protocol.TokenizeResponse")
        return JSONResponse(content=generator.model_dump())
    

@router.post("/v1/decode", response_model=functional_schema.ResponseDecode)
async def decode(request: Request,
    request_type: Annotated[
        functional_schema.RequestDecode,
        Body(openapi_examples=request_openapi_examples["decode"])] = None
    ) -> Response:
    """Decodes token ids

    The request should be a JSON object with the following fields:
    - token_ids: The ids of the tokens
    - with_tokens_str : If the result should also include a list of str
    - vanilla (optional) : Whether we want the vanilla version of the tokenizers
    """
    model: Model = RESOURCES[RESOURCE_MODEL]
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


@router.post("/v2/decode", response_model=vllm_protocol.DetokenizeResponse)
@with_cancellation
async def decode_v2(request :Annotated[
        vllm_protocol.DetokenizeRequest,
        Body(openapi_examples=request_openapi_examples["vllm_decode"])],
        raw_request: Request
    ):
    """Decodes token ids

    The request should be a JSON object with the following fields:
    - tokens: The ids of the tokens
    - model : ID of the model to use
    """
    model: Model = RESOURCES[RESOURCE_MODEL]
    generator = await model.openai_serving_tokenization.create_detokenize(request, raw_request)
    if isinstance(generator, vllm_protocol.ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        if not isinstance(generator, vllm_protocol.DetokenizeResponse):
            raise TypeError("Expected generator to be an instance of vllm_protocol.DetokenizeResponse")
        return JSONResponse(content=generator.model_dump())


@router.post("/v1/split_text", response_model=functional_schema.ResponseSplitText)
async def split_text(request: Request,
    request_type: Annotated[
        functional_schema.RequestSplitText,
        Body(openapi_examples=request_openapi_examples["split_text"])] = None
    ):
    """Splits a text with a minimal number of token in each chunk. Each chunk is delimited by a separator

    The request should be a JSON object with the following fields:
    - text: The text to split
    - num_tokens_in_chunk (optional): The minimal number of tokens we want in each chunk
    - separators (optional) : The allowed separators between the chunks
    """
    model: Model = RESOURCES[RESOURCE_MODEL]

    request_dict = await request.json()
    split_text = model.split_text(**request_dict)
    response = {"split_text": split_text}

    return JSONResponse(response)


@router.post("/v1/metadata_text", response_model=functional_schema.ResponseMetadata)
async def metadata_text(request: Request,
    request_type: Annotated[
        functional_schema.RequestMetadata,
        Body(openapi_examples=request_openapi_examples["metadata_text"])] = None):
    """Gives meta data on a text

    The request should be a JSON object with the following fields:
    - text: The text to parse
    - truncation_side (optional): The truncation side of the tokenizer
    - max_length (optional) : The max length before truncation

    The default values for truncation_side and max_length are those of the underlying model
    """
    model: Model = RESOURCES[RESOURCE_MODEL]

    request_dict = await request.json()

    tokens_ids = model.tokenize(request_dict['text'])
    truncated_text = model.extract_text_outside_truncation(**request_dict)
    ret = {"tokens_nb": len(tokens_ids), "truncated_text": truncated_text}

    return JSONResponse(ret)


@router.post("/v1/chat/completions", response_model=Union[vllm_protocol.ErrorResponse, 
                                                          functional_schema.HappyvllmChatCompletionResponse])
@with_cancellation
async def create_chat_completion(request: Annotated[vllm_protocol.ChatCompletionRequest, Body(openapi_examples=request_openapi_examples["chat_completions"])],
                                 raw_request: Request):
    """Open AI compatible chat completion. See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html for more details
    """
    model: Model = RESOURCES[RESOURCE_MODEL]
    handler = model.openai_serving_chat
    if handler is None:
        raise HTTPException(
            status_code=400, 
            detail=f"The model does not support Chat Completions API"
        )
    verify_request(request)
    generator = await handler.create_chat_completion(
        request, raw_request)
    if isinstance(generator, vllm_protocol.ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, # type: ignore
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump()) # type: ignore


@router.post("/v1/completions", response_model=Union[vllm_protocol.ErrorResponse, 
                                                     functional_schema.HappyvllmCompletionResponse])
@with_cancellation
async def create_completion(request: Annotated[vllm_protocol.CompletionRequest, Body(openapi_examples=request_openapi_examples["completions"])],
                            raw_request: Request):
    """Open AI compatible completion. See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html for more details
    """
    model: Model = RESOURCES[RESOURCE_MODEL]
    handler = model.openai_serving_completion
    if handler is None:
        return base(raw_request, model).create_error_response(
            message="The model does not support Completions API")
    verify_request(request)
    generator = await handler.create_completion(
        request, raw_request)
    if isinstance(generator, vllm_protocol.ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.post("/v1/embeddings", response_model=Union[vllm_protocol.ErrorResponse, vllm_protocol.EmbeddingResponse])
@with_cancellation
async def create_embedding(request: vllm_protocol.EmbeddingRequest, raw_request: Request):
    model: Model = RESOURCES[RESOURCE_MODEL]
    handler = model.openai_serving_embedding
    if handler is None:
        return base(raw_request, model).create_error_response(
                message="The model does not support Embeddings API")
    generator = await handler.create_embedding(request, raw_request)

    if isinstance(generator, vllm_protocol.ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, vllm_protocol.EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/abort_request")
async def abort_request(request: functional_schema.RequestAbortRequest):
    model: Model = RESOURCES[RESOURCE_MODEL]
    model._model.engine.abort_request(request.request_id)


if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:

    @router.post("/v1/load_lora_adapter")
    async def load_lora_adapter(request: vllm_protocol.LoadLoraAdapterRequest):
        model: Model = RESOURCES[RESOURCE_MODEL]
        response = await model.openai_serving_chat.load_lora_adapter(request)
        if isinstance(response, vllm_protocol.ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        response = await model.openai_serving_completion.load_lora_adapter(request)
        if isinstance(response, vllm_protocol.ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @router.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(request: vllm_protocol.UnloadLoraAdapterRequest):
        model: Model = RESOURCES[RESOURCE_MODEL]
        response = await model.openai_serving_chat.unload_lora_adapter(request)
        if isinstance(response, vllm_protocol.ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        response = await model.openai_serving_completion.unload_lora_adapter(request)
        if isinstance(response, vllm_protocol.ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)


    