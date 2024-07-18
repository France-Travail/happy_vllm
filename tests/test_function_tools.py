import pytest
from argparse import Namespace

from happy_vllm.function_tools.functions import ToolFunctions, Weather, Music, get_tools, update_tools, clean_tools, get_tools_prompt, reset_tools_dict_and_tools


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


def test_update_tools():
    # Test get_tools function with good arguments
    reset_tools_dict_and_tools()
    args = Namespace(tools=['weather', 'music'], tool_choice=['weather'])
    update_tools(args)
    TOOLS_DICT, TOOLS = get_tools()
    assert 'weather' in TOOLS_DICT
    assert 'music' not in TOOLS_DICT
    assert isinstance(TOOLS_DICT['weather'], Weather)
    assert TOOLS == ['weather']

    # Test get_tools function with wrong arguments
    reset_tools_dict_and_tools()
    args = Namespace(tools=['weather', 'music'], tool_choice='none')
    update_tools(args)
    TOOLS_DICT, TOOLS = get_tools()
    assert TOOLS_DICT is None
    assert TOOLS is None

    reset_tools_dict_and_tools()
    args = Namespace(tools=['weather', 'music'], tool_choice=['INVALID'])
    with pytest.raises(KeyError, match="The tool 'invalid' is not available in TOOLS_DICT"):
        update_tools(args)
    
    reset_tools_dict_and_tools()
    args = Namespace(tools=None, tool_choice=['weather'])
    with pytest.raises(ValueError, match="The argument '--tools' is required when '--tool-choice' is specified"):
        update_tools(args)

    reset_tools_dict_and_tools()
    args = Namespace(tools=['weather', 'music'], tool_choice=None)
    with pytest.raises(ValueError, match="The argument '--tool-choice' is required when '--tools' is specified"):
        update_tools(args)


def test_clean_tools():
    reset_tools_dict_and_tools()
    clean_tools()
    TOOLS_DICT, TOOLS = get_tools()
    assert TOOLS_DICT == {}


def test_get_tools_prompt():
    reset_tools_dict_and_tools()
    # Test get_tools_prompt function with good arguments
    args = Namespace(tools=['weather', 'music'], tool_choice=['weather'])
    update_tools(args)
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