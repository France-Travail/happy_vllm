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
import asyncio
import uvicorn
import argparse

from vllm.entrypoints.launcher import serve_http 
import vllm.entrypoints.openai.api_server as vllm_api_server


from happy_vllm.utils_args import parse_args
from happy_vllm.rpc.server import run_rpc_server
from happy_vllm.application import declare_application


TIMEOUT_KEEP_ALIVE = 5 # seconds

def main(**uvicorn_kwargs) -> None:
    args = parse_args()
    asyncio.run(launch_app(args, **uvicorn_kwargs))
    

def happy_vllm_build_async_engine_client(args):
    """Replace vllm.entrypoints.openai.api_server.run_rpc_server by happy_vllm.run_rpc_server
    """
    vllm_api_server.run_rpc_server  = run_rpc_server
    return vllm_api_server.build_async_engine_client(args)


async def launch_app(args, **uvicorn_kwargs):
    async with happy_vllm_build_async_engine_client(args) as async_engine_client:
        app = await declare_application(async_engine_client, args=args)
        shutdown_task = await serve_http(app,
                                        engine=async_engine_client,
                                        host=args.host,
                                        port=args.port,
                                        log_level=args.uvicorn_log_level,
                                        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                                        ssl_keyfile=args.ssl_keyfile,
                                        ssl_certfile=args.ssl_certfile,
                                        ssl_ca_certs=args.ssl_ca_certs,
                                        ssl_cert_reqs=args.ssl_cert_reqs,
                                        **uvicorn_kwargs)
    await shutdown_task

if __name__ == "__main__":
    main()