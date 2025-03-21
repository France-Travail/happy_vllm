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
import logging

from vllm.config import VllmConfig
from prometheus_client import Gauge
from vllm.usage.usage_lib import UsageContext
from vllm.engine.multiprocessing.engine import MQLLMEngine


logger = logging.getLogger(__name__)


def run_mp_engine(
        vllm_config: VllmConfig, usage_context: UsageContext,
        ipc_path: str, disable_log_stats: bool,
        disable_log_requests: bool, engine_alive):
    
    try :
        def signal_handler(*_) -> None:
            # Interrupt server on sigterm
            raise KeyboardInterrupt("MQLLMEngine terminated")
        
        engine = MQLLMEngine.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_stats=disable_log_stats,
            disable_log_requests=disable_log_requests,
            ipc_path=ipc_path)
        model_consumed_memory = Gauge("model_memory_usage", "Model Consumed GPU Memory in GB ")
        if vllm_config.scheduler_config.num_scheduler_steps > 1 :
            model_consumed_memory.set(round(engine.engine.model_executor.driver_worker.model_runner._base_model_runner.model_memory_usage/float(2**30),2)) # type: ignore
        else:
            model_consumed_memory.set(round(engine.engine.model_executor.driver_worker.model_runner.model_memory_usage/float(2**30),2)) # type: ignore
        signal.signal(signal.SIGTERM, signal_handler)
        engine.start()
    except BaseException as e:
        logger.exception(e)
        engine_alive.value = False
        raise e
     