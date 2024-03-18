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

import uvicorn
import argparse
from vllm.engine.arg_utils import AsyncEngineArgs

from happy_vllm.application import declare_application


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='127.0.0.1')
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_name", type=str, default='?')
    parser.add_argument("--explicit_errors", action='store_true')
    parser = AsyncEngineArgs.add_cli_args(parser)
    cli_args = parser.parse_args()

    app = declare_application(cli_args=cli_args)

    uvicorn.run(app,
                host=cli_args.host,
                port=cli_args.port)

if __name__ == "__main__":
    main()