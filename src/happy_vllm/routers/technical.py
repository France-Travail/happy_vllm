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


from typing import Annotated
from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from ..core.resources import RESOURCES, RESOURCE_MODEL
from ..model.model_base import Model

from happy_vllm import utils
from happy_vllm.routers.schemas import technical as technical_schema

# Technical router
router = APIRouter()


@router.get(
    "/liveness",
    response_model=technical_schema.ResponseLiveness,
    name="liveness",
    tags=["technical"],
)
async def get_liveness() -> technical_schema.ResponseLiveness:
    """Liveness probe for k8s"""
    liveness_msg = technical_schema.ResponseLiveness(alive="ok")
    return liveness_msg


@router.get(
    "/readiness",
    response_model=technical_schema.ResponseReadiness,
    name="readiness",
    tags=["technical"],
)
async def get_readiness() -> technical_schema.ResponseReadiness:
    """Readiness probe for k8s"""
    model: Model = RESOURCES.get(RESOURCE_MODEL)

    if model and model.is_model_loaded():
        return technical_schema.ResponseReadiness(ready="ok")
    else:
        return technical_schema.ResponseReadiness(ready="ko")


@router.get(
    "/v1/info",
    response_model=technical_schema.ResponseInformation,
    name="information",
    tags=["technical"],
)
async def info() -> technical_schema.ResponseInformation:
    """Rest resource for info"""
    model: Model = RESOURCES.get(RESOURCE_MODEL)

    return technical_schema.ResponseInformation(
        application=model.app_name,
        version=utils.get_package_version(),
        model_name=model._model_conf.get("model_name", "?"),
        vllm_version=utils.get_vllm_version(),
        truncation_side=model.original_truncation_side,
        max_length=model.max_model_len
    )


@router.get("/v1/models", response_model=technical_schema.HappyvllmModelList)
async def show_available_models():
    model: Model = RESOURCES.get(RESOURCE_MODEL)
    models = await model.openai_serving_completion.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/v1/launch_arguments")
async def launch_arguments():
    model: Model = RESOURCES.get(RESOURCE_MODEL)
    launch_arguments = model.launch_arguments
    return JSONResponse(content=launch_arguments)