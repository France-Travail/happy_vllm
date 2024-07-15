import logging
from copy import copy
from typing import Union
from argparse import Namespace


class ToolFunctions:
    def __init__(self, description:Union[str, None], parameters:Union[dict, None], name:Union[str, None], tool_type:Union[str, None]):
        self.description:str = description
        self.parameters:dict = parameters
        self.name:str = name
        self.tool_type:str = tool_type
        self._check_attributes()

    def _check_attributes(self):
        if not self.description:
            raise NotImplementedError("This attributes must be different to None")
        if not self.parameters:
            raise NotImplementedError("This attributes must be different to None")
        if not self.name:
            raise NotImplementedError("This attributes must be different to None")
        if not self.tool_type:
            raise NotImplementedError("This attributes must be different to None")

    def generate_dict(self):
        return {
            "type": self.tool_type,
            "function": {
                "name": self.name,         
                "description": self.description,
                "parameters": self.parameters,
                
            }
        } 


class Weather(ToolFunctions):
    def __init__(self):
        tool_type = "function"
        name = "get_current_weather"
        description = "Get current weather"
        parameters = {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            }
        super().__init__(description=description, parameters=parameters, name=name, tool_type=tool_type)


class Music(ToolFunctions):
    def __init__(self):
        tool_type = "function"
        name = "ask_database"
        description = "Use this function to answer user questions about music. Input should be a fully formed SQL query."
        parameters = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
                                <schema>
                                The query should be returned in plain text, not in JSON.
                                """,
                    }
                },
                "required": ["query"],
            }
        super().__init__(description=description, parameters=parameters, name=name, tool_type=tool_type)


TOOLS_DICT = {
    'weather': Weather,
    'music': Music
}
TOOLS = []


def get_tools(args: Namespace):
    global TOOLS_DICT
    global TOOLS
    if args.tools and 'none' not in args.tool_choice:
        tools = {}
        for t in args.tool_choice:
            tools[t.lower()] = TOOLS_DICT[t.lower()]()
        TOOLS_DICT = tools
        TOOLS = [t.lower() for t in args.tool_choice]
    else:
        TOOLS_DICT = None
        TOOLS = None


def clean_tools():
    global TOOLS_DICT
    TOOLS_DICT.clear()


def get_tools_prompt() -> dict:
    tools_dict = copy(TOOLS_DICT)
    tools = copy(TOOLS)
    if tools:
        return {
            "tools": [tools_dict[t].generate_dict() for t in tools],
            "tool_choice": [
                {
                    "type": tools_dict[t].tool_type, 
                    "function": {"name":tools_dict[t].name}
                }
                for t in tools
            ][0]
        }
    else:
        return None