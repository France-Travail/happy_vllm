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


"""Technical schemas"""
import os
import json
from pydantic import BaseModel, Field

from vllm.entrypoints.openai.protocol import ModelList

# Load the response examples
directory = os.path.dirname(os.path.abspath(__file__))
response_examples_path = os.path.join(directory, "examples", "response.json")
with open(response_examples_path, 'r') as file:
    response_examples = json.load(file)


class ResponseLiveness(BaseModel):
    """Return object for liveness probe"""

    alive: str = Field(None, title="Alive message")
    model_config = {
        "json_schema_extra": {
            "examples": [
                response_examples["liveness"]
            ]
        }
    }

class ResponseReadiness(BaseModel):
    """Return object for readiness probe"""

    ready: str = Field(None, title="Ready message")
    model_config = {
        "json_schema_extra": {
            "examples": [
                response_examples["readiness"]
            ]
        }
    }


class ResponseInformation(BaseModel):
    """Return object for info resource"""

    application: str = Field(None, title="Application name")
    version: str = Field(None, title="Application version")
    vllm_version: str = Field(None, title="Version of vLLM")
    model_name: str = Field(None, title="Model name")
    truncation_side: str = Field(None, title="Truncation side")
    max_length : int = Field(None, title="Max length")
    model_config = {
        "json_schema_extra": {
            "examples": [
                response_examples["information"]
            ]
        }
    }


class ResponseLiveMetrics(BaseModel):
    requests_running: int = Field(None, title="Number of running requests")
    requests_swapped: int = Field(None, title="Number of swapped requests")
    requests_pending: int = Field(None, title="Number of pending requests")
    gpu_cache_usage: float = Field(None, title="GPU cache usage")
    cpu_cache_usage: float = Field(None, title="CPU cache usage")
    model_config = {
        "json_schema_extra": {
            "examples": [
                response_examples["live_metrics"]
            ]
        }
    }


class HappyvllmModelList(ModelList):
    model_config = {"json_schema_extra": {"examples": [response_examples["models"]]}}


