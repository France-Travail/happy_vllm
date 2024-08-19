import asyncio
import cloudpickle

from typing import Any, Coroutine
from prometheus_client import Gauge
from typing_extensions import Never
from vllm.usage.usage_lib import UsageContext
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.rpc.server import AsyncEngineRPCServer, run_server
from vllm.entrypoints.openai.rpc import RPCAbortRequest, RPCGenerateRequest, RPCUtilityRequest


class CustomAsyncEngineRPCServer(AsyncEngineRPCServer):

    def __init__(
            self, 
            async_engine_args: AsyncEngineArgs,
            usage_context: UsageContext,
            port: int
        ):
        super().__init__(async_engine_args=async_engine_args, usage_context=usage_context, port=port)
            

    # async def get_model_memory_usage(self, identity):
    #     model_consumed_memory = round(self.engine.engine.model_executor.driver_worker.model_runner.model_memory_usage/float(2**30),2) # type: ignore
    #     await self.socket.send_multipart(
    #         [identity, cloudpickle.dumps(model_consumed_memory)])
    

    # def _make_handler_coro(self, identity,message) -> Coroutine[Any, Any, Never]:
    #     request = cloudpickle.loads(message)

    #     if isinstance(request, RPCGenerateRequest):
    #         return super()._make_handler_coro(identity, message)

    #     elif isinstance(request, RPCAbortRequest):
    #         return super()._make_handler_coro(identity, message)
        
    #     elif isinstance(request, RPCUtilityRequest):
    #         if request == RPCUtilityRequest.GET_MODEL_CONFIG:

    #         else:
    #             return super()._make_handler_coro(identity, message)
    #     else:
    #         raise ValueError(f"Unknown RPCRequest type: {request}")
        

def run_rpc_server(async_engine_args: AsyncEngineArgs,
                   usage_context: UsageContext, port: int):
    server = CustomAsyncEngineRPCServer(async_engine_args, usage_context, port)
    model_consumed_memory = Gauge("model_memory_usage", "Model Consumed GPU Memory in GB ")
    model_consumed_memory.set(round(server.engine.engine.model_executor.driver_worker.model_runner.model_memory_usage/float(2**30),2)) # type: ignore
    asyncio.run(run_server(server))
     