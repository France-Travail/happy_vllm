# Functions calling

To avoid repeatedly specifying all attributes related to tools and tool_choice when functions calling, you can create a classe inherit ```functions.ToolFunctions```.

You need to instantiate 4 attributes: 
 - description (string)
 - parameters (dict)
 - name (string)
 - tool_type (string) 

Each attributes correpondings as [Openai api](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)