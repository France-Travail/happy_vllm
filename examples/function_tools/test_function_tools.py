import pytest
from argparse import Namespace

from functions import ToolFunctions, Weather, Music, get_tools, get_tools_prompt, reset_tools_dict_and_tools


def test_toolfunctions():
    # Test ToolFunctions class creation
    tf = ToolFunctions(description="Test tool", parameters={"param1": "value1"}, name="TestTool", tool_type="Utility")
    assert tf.description == "Test tool"
    assert tf.parameters == {"param1": "value1"}
    assert tf.name == "TestTool"
    assert tf.tool_type == "Utility"

    expected_output = {
        "type": "Utility",
        "function": {
            "name": "TestTool",
            "description": "Test tool",
            "parameters": {"param1": "value1"},
        }
    }
    assert tf.generate_dict() == expected_output

    # Test ToolFunctions class creation with missing parameters
    with pytest.raises(NotImplementedError, match="This attributes must be different to None"):
        ToolFunctions(description=None, parameters={"param1": "value1"}, name="TestTool", tool_type="Utility")
    with pytest.raises(NotImplementedError, match="This attributes must be different to None"):
        ToolFunctions(description="Test tool", parameters=None, name="TestTool", tool_type="Utility")
    with pytest.raises(NotImplementedError, match="This attributes must be different to None"):
        ToolFunctions(description="Test tool", parameters={"param1": "value1"}, name=None, tool_type="Utility")
    with pytest.raises(NotImplementedError, match="This attributes must be different to None"):
        ToolFunctions(description="Test tool", parameters={"param1": "value1"}, name="TestTool", tool_type=None)


def test_get_tools_prompt():
    reset_tools_dict_and_tools()
    # Test get_tools_prompt function with good arguments
    TOOLS_DICT, TOOLS = get_tools()
    expected_output = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get current weather",
                    "parameters": {
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
                }
            }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_current_weather"}
        }
    }
    assert get_tools_prompt() == expected_output