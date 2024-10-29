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

import signal
import uvloop

from prometheus_client import Gauge
from vllm.usage.usage_lib import UsageContext
from vllm import AsyncEngineArgs
from vllm.engine.multiprocessing.engine import MQLLMEngine


def run_mp_engine(engine_args: AsyncEngineArgs,
                   usage_context: UsageContext, ipc_path: str):
    def signal_handler(*_) -> None:
        # Interrupt server on sigterm
        raise KeyboardInterrupt("MQLLMEngine terminated")
    signal.signal(signal.SIGTERM, signal_handler)
    engine = MQLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=usage_context,
        ipc_path=ipc_path
    )
    model_consumed_memory = Gauge("model_memory_usage", "Model Consumed GPU Memory in GB ")
    if engine_args.num_scheduler_steps > 1 :
        model_consumed_memory.set(round(engine.engine.model_executor.driver_worker.model_runner._base_model_runner.model_memory_usage/float(2**30),2)) # type: ignore
    else:
        model_consumed_memory.set(round(engine.engine.model_executor.driver_worker.model_runner.model_memory_usage/float(2**30),2)) # type: ignore
    engine.start()
     