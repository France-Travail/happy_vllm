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

import os
import shutil
from .conftest import TEST_MODELS_DIR
from fastapi.testclient import TestClient

from happy_vllm import utils


def teardown_module():
    if os.path.exists(TEST_MODELS_DIR):
        if os.path.isdir(TEST_MODELS_DIR):
            shutil.rmtree(TEST_MODELS_DIR)


def test_get_liveness(test_base_client: TestClient):
    """Test the technical route /liveness"""
    response = test_base_client.get("/tests/liveness")
    assert response.status_code == 200
    assert response.json() == {"alive": "ok"}


def test_get_readiness_ko(test_base_client: TestClient):
    """Test the technical route /readiness when the model is not loaded"""
    response = test_base_client.get("/tests/readiness")
    assert response.status_code == 200
    assert response.json() == {"ready": "ko"}


def test_get_readiness_ok(test_complete_client: TestClient):
    """Test the technical route /readiness when the model is fully loaded"""
    response = test_complete_client.get("/tests/readiness")
    assert response.status_code == 200
    assert response.json() == {"ready": "ok"}


def test_info(test_complete_client: TestClient):
    """Test the technical route /v1/info"""
    response = test_complete_client.get("/tests/v1/info")
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["application"] == "APP_TESTS"
    assert response_json["version"] == utils.get_package_version()
    assert response_json["model_name"] == "TEST MODEL"
    assert response_json["truncation_side"] == "right"
    assert response_json["max_length"] == 2048


def test_launch_arguments(test_complete_client: TestClient):
    """Test the technical route /v1/launch_arguments"""
    response = test_complete_client.get("/tests/v1/launch_arguments")
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["app_name"] == "APP_TESTS"
    assert response_json["model_name"] == "TEST MODEL"
    assert response_json["with_launch_arguments"] == True