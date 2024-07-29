# Functions calling

## Deploy a new function
To avoid repeatedly specifying all attributes related to tools and tool_choice when using functions calling, you can create a class inheriting from ```functions.ToolFunctions```. 

You need to instantiate 4 attributes: 
 - description (string)
 - parameters (dict)
 - name (string)
 - tool_type (string) 

Each attributes corresponding to [Openai api](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)

After the class is created, you have to declare it in TOOLS_DICT and TOOLS global variables in ```routers.shcema.functional``` (add weather function tool for example).

```
TOOLS_DICT = {
    'weather': Weather(),
}
TOOLS = ['weather']
```

**You can only use one function by deployement**. 

## Called functions
To use this implementation, you must replace the original ```routers.function.py``` and ```routers.schema.function.py``` files with the respective files in the folder ```example.function_tools```.


## How to use
After deploying your REST API, you just need to call it with the following route ```/v1/chat/completions_tools``` and a body as follows:

```
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Who won the world series in 2020?"
    },
    {
      "role": "assistant",
      "content": "The Los Angeles Dodgers won the World Series in 2020."
    },
    {
      "role": "user",
      "content": "How was the weather ?"
    }
  ],
  "model": "my_model"
}
```

The tool will be automatically called

## Warning

From vllm 0.5.0 to 0.5.3.post1, tool_choice's option ```auto``` and ```required``` are not yet implemented.

