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


"""Test configuration

This conftest.py file is the first loaded by pytest when the tests are executed,
see the documentation for more infos :
https://docs.pytest.org/en/7.2.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files

We use it to :
- create a test model
- create a AsyncClient named test_base_client which does not load the test model
- create a AsyncClient named test_complete_client which does load the test model
- set some environment variables to check if they are well set in the app settings

> More details about test_base_client and test_complete_client:
>
> By default a AsyncClient does not triggered lifespan so the startup and shutdown events are never fired
> and the model is never loaded.
> We can fire thoses events by using AsyncClient as a context manager so we use two AsyncClient in
> our tests : a test_base_client that does not load the model and test_complete_client that does
> load the model thanks to a context manager. It allows us to test the behavior of our application
> when a model is not loaded (which should not happen).
>
> AsyncClient need to use ASGITransport to be sure the routes are available and avoid 404 response
> As AsyncClient don't trigger the lifespan by default in httpx, we have to use asgi_lifespan package
> (source httpx.AsyncClient doc) to init a LifespanManager and trigger the lifespan
"""

import os
import pytest
import pytest_asyncio
from typing import Union
from pathlib import Path
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport

from argparse import Namespace
from pydantic_settings import BaseSettings, SettingsConfigDict


# Manage the huggingface token
class HuggingfaceSettings(BaseSettings):
    """A class to get the HuggingFace token
    """
    hf_token : Union[str, None] = None
    model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))

huggingface_settings = HuggingfaceSettings()
os.environ['HF_TOKEN'] = huggingface_settings.hf_token

# Set paths
TEST_DIR = Path(__file__).parent.resolve()
TEST_DATA_DIR = TEST_DIR / "data"
TEST_MODELS_DIR = TEST_DATA_DIR / "models"

# Set environment variables for testing
os.environ["app_name"] = "APP_TESTS"
os.environ["api_endpoint_prefix"] = "/tests"
os.environ["MODEL_NAME"] = "TEST MODEL"
os.environ["MODEL"] = "test"
os.environ["TEST_MODELS_DIR"] = str(TEST_MODELS_DIR)
os.environ["TEST_MODE"] = str(True)

# We must import the utils module after setting the environnement variables because
# it also imports the .core folder via the __init__ and it may impact the other tests
from happy_vllm import utils
os.environ["tokenizer_name"] = utils.TEST_TOKENIZER_NAME

from happy_vllm.core import resources
from happy_vllm.model.model_base import Model
from happy_vllm.application import declare_application
from happy_vllm.launch import happy_vllm_build_async_engine_client


@pytest_asyncio.fixture(scope="session")
async def test_base_client() -> AsyncClient:
    """Basic AsyncClient that do not run startup and shutdown events"""
    args = Namespace(
        explicit_errors=False,
        model_name=os.environ['MODEL_NAME'],
        model=os.environ['MODEL'],
        app_name=os.environ["app_name"],
        api_endpoint_prefix=os.environ["api_endpoint_prefix"],
        allow_credentials=True,
        allowed_origins=["*"],
        allowed_methods=["*"],
        allowed_headers=["*"],
        root_path=None,
        with_launch_arguments=True,
        disable_fastapi_docs=False,
        enable_server_load_tracking=False,
        extra_information=str(TEST_DATA_DIR / "extra_information.json")
    )
    app = await declare_application(happy_vllm_build_async_engine_client(args), args)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True)


@pytest_asyncio.fixture()
async def test_complete_client(monkeypatch) -> AsyncClient:
    """Complete AsyncClient that do run startup and shutdown events to load
    the model
    """
    # Use base model for tests
    monkeypatch.setattr(resources, "Model", Model)
    args = Namespace(
        explicit_errors=False,
        model_name=os.environ['MODEL_NAME'],
        model=os.environ['MODEL'],
        app_name=os.environ["app_name"],
        api_endpoint_prefix=os.environ["api_endpoint_prefix"],
        allow_credentials=True,
        allowed_origins=["*"],
        allowed_methods=["*"],
        allowed_headers=["*"],
        root_path=None,
        with_launch_arguments=True,
        disable_fastapi_docs=False,
        enable_server_load_tracking=False,
        extra_information=str(TEST_DATA_DIR / "extra_information.json")
    )
    app = await declare_application(happy_vllm_build_async_engine_client(args), args)
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test", follow_redirects=True) as client:
            print(f"RESSOURCE : {resources.RESOURCES}")
            yield client