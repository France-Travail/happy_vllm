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

import uvloop

from prometheus_client import Gauge
from vllm.usage.usage_lib import UsageContext
from vllm import AsyncEngineArgs
from vllm.entrypoints.openai.rpc.server import AsyncEngineRPCServer, run_server


def run_rpc_server(async_engine_args: AsyncEngineArgs,
                   usage_context: UsageContext, rpc_path: str):
    server = AsyncEngineRPCServer(async_engine_args=async_engine_args, usage_context=usage_context, rpc_path=rpc_path)
    model_consumed_memory = Gauge("model_memory_usage", "Model Consumed GPU Memory in GB ")
    model_consumed_memory.set(round(server.engine.engine.model_executor.driver_worker.model_runner.model_memory_usage/float(2**30),2)) # type: ignore
    uvloop.run(run_server(server))
     