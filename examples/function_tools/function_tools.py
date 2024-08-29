from typing import Union


class FunctionTool:
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
    def __init__(self, description: Union[str, None], parameters: Union[dict, None], name: Union[str, None], tool_type: Union[str, None]):
        self.description: str = description
        self.parameters: dict = parameters
        self.name: str = name
        self.tool_type: str = tool_type
        self._check_attributes()

    def _check_attributes(self):
        if not self.description:
            raise AttributeError("This attributes must be different to None")
        if not self.parameters:
            raise AttributeError("This attributes must be different to None")
        if not self.name:
            raise AttributeError("This attributes must be different to None")
        if not self.tool_type:
            raise AttributeError("This attributes must be different to None")

    def generate_dict(self):
        return {
            "type": self.tool_type,
            "function": {
                "name": self.name,         
                "description": self.description,
                "parameters": self.parameters,
                
            }
        } 


class Weather(FunctionTool):
    """
    Represents an example tool function about the weather.
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