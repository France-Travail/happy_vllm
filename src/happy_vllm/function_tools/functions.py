import logging
from copy import copy
from typing import Union
from argparse import Namespace


class ToolFunctions:
    """
    Represents a tool function with specific attributes.

    Attributes:
    description (str): Description of the tool function.
    parameters (dict): Parameters required for the tool function.
    name (str): Name of the tool function.
    tool_type (str): Type of the tool function.

    Methods:
    __init__(description: Union[str, None], parameters: Union[dict, None], name: Union[str, None], tool_type: Union[str, None]):
        Initializes a ToolFunctions instance with the provided attributes. Raises NotImplementedError if any attribute is None.
    
    _check_attributes():
        Checks if the required attributes (description, parameters, name, tool_type) are not None.
        Raises NotImplementedError if any attribute is None.

    generate_dict() -> dict:
        Generates and returns a dictionary representation of the tool function, including its type, name, description, and parameters.
    """
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
    """
    Represents a example tool function about the weather.
    """
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
                "required": ["location", "format"]
            }
        super().__init__(description=description, parameters=parameters, name=name, tool_type=tool_type)


class Music(ToolFunctions):
    """
    Represents a example tool function about the music.
    """
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
                "required": ["query"]
            }
        super().__init__(description=description, parameters=parameters, name=name, tool_type=tool_type)


TOOLS_DICT = {
    'weather': Weather,
    'music': Music
}
TOOLS = []

def get_tools():
    return TOOLS_DICT, TOOLS

def reset_tools_dict_and_tools():
    """
    Resets the global variables TOOLS_DICT and TOOLS with new default values.

    Returns:
    tuple: A tuple containing the updated values of TOOLS_DICT and TOOLS. 
           TOOLS_DICT is a dictionary with keys for different tools and corresponding values, 
           and TOOLS is an empty list.
    """
    global TOOLS_DICT
    global TOOLS
    TOOLS_DICT = {
        'weather': Weather,
        'music': Music
    }
    TOOLS = []


def update_tools(args: Namespace):
    """
    Updates the global variables TOOLS_DICT and TOOLS based on the provided arguments.

    Args:
    args (Namespace): A Namespace object containing parsed command-line arguments.

    Returns:
    tuple: A tuple containing the updated values of TOOLS_DICT and TOOLS. 
           TOOLS_DICT is updated with instances of selected tools or set to None if no tools are selected.
           TOOLS is updated with names of selected tools or set to None if no tools are selected.
    """
    global TOOLS_DICT
    global TOOLS
    if args.tools and args.tool_choice and 'none' not in args.tool_choice:
        tools = {}
        for t in args.tool_choice:
            if TOOLS_DICT.get(t.lower(), None):
                tools[t.lower()] = TOOLS_DICT[t.lower()]()
            else:
                raise KeyError(f"The tool '{t.lower()}' is not available in TOOLS_DICT")
        TOOLS_DICT = tools
        TOOLS = [t.lower() for t in args.tool_choice]
    else:
        if args.tools and args.tool_choice is None:
            raise ValueError("The argument '--tool-choice' is required when '--tools' is specified")
        elif args.tools is None and args.tool_choice:
            raise ValueError("The argument '--tools' is required when '--tool-choice' is specified")
        TOOLS_DICT = None
        TOOLS = None


def clean_tools():
    """
    Clears the global variable TOOLS_DICT, removing all entries.

    Returns:
    dict: An empty dictionary representing the cleaned TOOLS_DICT after removal of all entries.
    """
    global TOOLS_DICT
    if TOOLS_DICT:
        TOOLS_DICT.clear()


def get_tools_prompt() -> dict:
    """
    Returns a dictionary containing information about selected tools.

    Returns:
    dict or None: A dictionary containing information about selected tools, structured as follows:
                  - "tools": A list of dictionaries, each representing a tool's generated dictionary.
                  - "tool_choice": A dictionary containing type and function details of the first tool in the list,
                                   or None if TOOLS is empty.
                  Returns None if TOOLS is empty.
    """
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