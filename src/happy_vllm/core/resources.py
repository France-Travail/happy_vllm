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


"""Resources for the FastAPI application

This module define resources that need to be instantiated at startup in a global
variable resources that can be used in routes.

This is the way your machine learning models can be loaded in memory at startup
so they can handle requests.
"""

import logging
from typing import Callable
from argparse import Namespace
from contextlib import asynccontextmanager

from fastapi import FastAPI
from ..model.model_base import Model

from vllm.entrypoints.openai.rpc.client import AsyncEngineRPCClient


logger = logging.getLogger(__name__)

RESOURCES = {}
RESOURCE_MODEL = "model"

def get_lifespan(async_engine_client: AsyncEngineRPCClient, args: Namespace) -> Callable:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the ML model
        model = Model(app_name=args.app_name)
        await model.loading(async_engine_client, args=args)
        logger.info("Model loaded")

        RESOURCES[RESOURCE_MODEL] = model

        # Force log once to have informations in /metrics before requests
        await model._model.do_log_stats()

        yield

        # Clean up the ML models and release the resources
        RESOURCES.clear()
    return lifespan
