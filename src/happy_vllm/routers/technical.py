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


from fastapi import APIRouter
from starlette.responses import JSONResponse

from ..core.config import settings
from ..core.resources import RESOURCES, RESOURCE_MODEL
from ..model.model_base import Model
from .schemas.technical import ResponseInformation, ResponseLiveness, ResponseReadiness

# Technical router
router = APIRouter()


@router.get(
    "/liveness",
    response_model=ResponseLiveness,
    name="liveness",
    tags=["technical"],
)
async def get_liveness() -> ResponseLiveness:
    """Liveness probe for k8s"""
    liveness_msg = ResponseLiveness(alive="ok")
    return liveness_msg


@router.get(
    "/readiness",
    response_model=ResponseReadiness,
    name="readiness",
    tags=["technical"],
)
async def get_readiness() -> ResponseReadiness:
    """Readiness probe for k8s"""
    model: Model = RESOURCES.get(RESOURCE_MODEL)

    if model and model.is_model_loaded():
        return ResponseReadiness(ready="ok")
    else:
        return ResponseReadiness(ready="ko")


@router.get(
    "/info",
    response_model=ResponseInformation,
    name="information",
    tags=["technical"],
)
async def info() -> ResponseInformation:
    """Rest resource for info"""
    model: Model = RESOURCES.get(RESOURCE_MODEL)

    return ResponseInformation(
        application=settings.app_name,
        version=settings.app_version,
        model_name=model._model_conf.get("model_name", "?"),
        truncation_side=model.original_truncation_side,
        max_length=model.max_model_len
    )


@router.get("/live_metrics")
async def get_live_metrics() -> JSONResponse:
    model: Model = RESOURCES.get(RESOURCE_MODEL)

    gpu_cache_usage = model.get_gpu_kv_cache_usage()
    cpu_cache_usage = model.get_cpu_kv_cache_usage()
    metrics = model.get_status_requests()
    metrics["gpu_cache_usage"] = gpu_cache_usage
    metrics["cpu_cache_usage"] = cpu_cache_usage
    return JSONResponse(metrics)