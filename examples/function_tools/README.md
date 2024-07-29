# Functions calling

## Deploy a new function
To avoid repeatedly specifying all attributes related to tools and tool_choice when using functions calling, you can create a classe inheriting from ```functions.ToolFunctions```.

You need to instantiate 4 attributes: 
 - description (string)
 - parameters (dict)
 - name (string)
 - tool_type (string) 

Each attributes correpondings to [Openai api](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)

After the class is created, you have to declare in TOOLS_DICT and TOOLS global variables (add weather function tool for example).

```
TOOLS_DICT = {
    'weather': Weather(),
}
TOOLS = ['weather']
```

## Called functions
To use this implementation, you must replace ```routers.function.py``` and ```routers.schema.function.py``` with the respective files in the folder ```example.function_tools```.
After deploying your REST API, you can call it with the following route ```/v1/chat/completions_tools```

## To know

From vllm 0.5.0 to 0.5.3.post1, tool_choice's option ```auto``` and ```required``` are not yet implemented. You can only use one function by deployement. Provide a complete body (asking for the temperature for example) : 

```
{
    "type": "function",
    "function": {"name": "get_current_weather"}
}
```

