import pytest

from starlette.requests import Request
from unittest.mock import patch, MagicMock
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionToolsParam, ChatCompletionNamedToolChoiceParam


from happy_vllm.routers.schemas.functional import update_chat_completion_request


TOOLS = [
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
}]
TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "get_current_weather"}
}


@pytest.mark.asyncio
async def test_update_chat_completion_request():
    with patch('happy_vllm.routers.schemas.functional.get_tools_prompt')as mock_get_tools_prompt:
        mock_get_tools_prompt.return_value = {
            "tools": TOOLS,
            "tool_choice": TOOL_CHOICE
        }
        messages = [
            {
            "role": "system",
            "content": "You are a helpful assistant."
            },
            {
            "role": "user",
            "content": "How was the weather in Paris"
            }
        ]
        
        mock_request = MagicMock(spec=Request)
        data = ChatCompletionRequest(messages=messages, model="my_model")
        updated_data = await update_chat_completion_request(mock_request, data)
        assert updated_data.tools == [ChatCompletionToolsParam(**t) for t in TOOLS]
        assert updated_data.tool_choice == ChatCompletionNamedToolChoiceParam(**TOOL_CHOICE)
        assert updated_data.top_logprobs is None