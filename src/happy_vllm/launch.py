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
import asyncio
import logging
import uvicorn
import argparse

from vllm.entrypoints.launcher import serve_http 
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import is_valid_ipv6_address, set_ulimit
import vllm.entrypoints.openai.api_server as vllm_api_server
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args
from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager

from happy_vllm.utils_args import parse_args
from happy_vllm.engine.mp_engine import run_mp_engine
from happy_vllm.application import declare_application


TIMEOUT_KEEP_ALIVE = 5 # seconds
logger = logging.getLogger(__name__)


def main(**uvicorn_kwargs) -> None:
    args = parse_args()
    asyncio.run(launch_app(args, **uvicorn_kwargs))
    

def happy_vllm_build_async_engine_client(args):
    """Replace vllm.entrypoints.openai.api_server.run_rpc_server by happy_vllm.run_rpc_server
    """
    vllm_api_server.run_mp_engine  = run_mp_engine
    return vllm_api_server.build_async_engine_client(args)


async def launch_app(args, **uvicorn_kwargs):
    # Check args
    validate_parsed_serve_args(args)
    # Register new tool parser
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    valide_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valide_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valide_tool_parses)} }})")
    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.enable_reasoning \
        and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})")

    # Bind socket
    sock_addr = (args.host or "", args.port)
    sock = vllm_api_server.create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")
    signal.signal(signal.SIGTERM, signal_handler)

    # Launch app
    async with happy_vllm_build_async_engine_client(args) as async_engine_client:
        app = await declare_application(async_engine_client, args=args)

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return '[' + a + ']'
            return a or "127.0.0.1"

        logger.info("Starting vLLM API server on http://%s:%d",
                    _listen_addr(sock_addr[0]), sock_addr[1])

        shutdown_task = await serve_http(app,
                                        sock=sock,
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

    sock.close()

    
if __name__ == "__main__":
    main()