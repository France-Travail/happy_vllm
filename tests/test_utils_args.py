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

from happy_vllm import utils_args


def test_get_parser():
    parser = utils_args.get_parser()
    application_settings = utils_args.ApplicationSettings()
    for key, value in application_settings.model_dump().items():
        assert parser.get_default(key) == value


def test_get_model_settings():
    parser = utils_args.get_parser()
    model_settings = utils_args.get_model_settings(parser)
    for key, field in model_settings.model_fields.items():
        if parser.get_default(key) is not None:
            assert parser.get_default(key) == field.default
            
    assert model_settings.model == os.environ['MODEL']
    assert model_settings.model_name == os.environ['MODEL_NAME']

