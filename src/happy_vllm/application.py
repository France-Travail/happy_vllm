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


from fastapi import FastAPI
from argparse import Namespace
from .routers import main_routeur
from .core.resources import get_lifespan
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware

from happy_vllm import utils
from happy_vllm.middlewares.exception import ExceptionHandlerMiddleware

def declare_application(args: Namespace) -> FastAPI:
    """Create the FastAPI application

    See https://fastapi.tiangolo.com/tutorial/first-steps/ to learn how to
    customize your FastAPI application
    """
    app = FastAPI(
        title=f"A REST API for vLLM",
        description=f"A REST API for vLLM, production ready",
        lifespan=get_lifespan(args=args),
        version=utils.get_package_version()
    )

    # Add prometheus asgi middleware to route /metrics requests
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # CORS middleware that allows all origins to avoid CORS problems
    # see https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    # Add exception middleware
    if args.explicit_errors:
        app.add_middleware(ExceptionHandlerMiddleware)

    #
    app.include_router(main_routeur, prefix=args.api_endpoint_prefix)

    app.root_path = args.root_path

    return app