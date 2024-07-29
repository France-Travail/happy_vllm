from copy import copy
from typing import Union
from function_tools import Weather

TOOLS_DICT = {
    'weather': Weather(),
}
TOOLS = ['weather']


def get_tools() -> tuple[dict, tuple]:
    """
    Returns a tuple containing TOOLS_DICT and TOOLS global object.
    """
    return TOOLS_DICT, TOOLS


def get_tools_prompt() -> Union[dict, None]:
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