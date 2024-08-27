import asyncio
import cloudpickle

from prometheus_client import Gauge
from typing_extensions import Never
from vllm.usage.usage_lib import UsageContext
from vllm import AsyncEngineArgs
from vllm.entrypoints.openai.rpc.server import AsyncEngineRPCServer, run_server


def run_rpc_server(async_engine_args: AsyncEngineArgs,
                   usage_context: UsageContext, port: int):
    server = AsyncEngineRPCServer(async_engine_args=async_engine_args, usage_context=usage_context, port=port)
    model_consumed_memory = Gauge("model_memory_usage", "Model Consumed GPU Memory in GB ")
    model_consumed_memory.set(round(server.engine.engine.model_executor.driver_worker.model_runner.model_memory_usage/float(2**30),2)) # type: ignore
    asyncio.run(run_server(server))
     