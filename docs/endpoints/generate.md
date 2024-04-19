# Generating endpoints

There are four endpoints used to generate content. The first two are direct copies of the endpoints provided by vLLM : `/v1/completions` and `/v1/chat/completions` and follow the Open AI contract. The last two `/v1/generate` and `/v1/generate_stream` are deprecated and should not be used anymore. We keep the relevant documentation until we delete them.

## Open AI compatible endpoints

For these two endpoints (`/v1/completions` and `/v1/chat/completions`) we refer you to the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). Some examples on how to use them are available in the swagger (whose adress is `127.0.0.1:5000/docs` by default)

## Deprecated generating endpoints

They have the same contract, the only difference is in the response.

The `/v1/generate` endpoint will give the whole response in one go whereas the `/v1/generate_stream` endpoint will provide a streaming response.

### Keywords

Here are the keywords you can send to the endpoint. The `prompt` keyword is the only one which is mandatory, all others are optional.

 - `prompt`: The prompt we want to send to the model to complete. Example : "Can you give me the seven wonders of the world ?" 
 - `prompt_in_response`: Whether we want the prompt in the response or no. Example : True (default value: False)
 - `response_pool`: A list of string. The model will be forced to answer one of these strings. Example : ["Yes", "No", "Maybe"] (default value: not activated). It is incompatible with the keyword `min_tokens`.
 - `json_format`: When used, specifify to the model that we want a json output and indicates the format of the json. It uses [LM-format-enforcer](https://github.com/noamgat/lm-format-enforcer). To have more details on how to fill this, see [the corresponding section below](#json-format)
 - `json_format_is_json_schema`: Indicates if the `json_format` is a simple json or a json schema. More details in [the corresponding section below](#json-format)

You can also add all the keywords permitted by vLLM such as `temperature`, `top_p`, `repetition_penalty`, ... (the whole list can be found [here](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)). 

Note that the keyword `logits_processors` is not allowed to be used since happy_vllm use this keyword to implement its own logits processors (and provide for example the `response_pool` or `json_format` keywords). 

If you would like to add a specific logits processor, feel free to open a PR or an issue.

### Output

The output is of the following form :

```
{
  "responses": [
    "My first response",
    "My second response"
  ],
  "finish_reasons": [
    "length",
    "stop"
  ],
  "usages": [{
      "prompt_tokens": 3,
      "completion_tokens": 7,
      "total_tokens": 10
    },
    {
      "prompt_tokens": 3,
      "completion_tokens": 5,
      "total_tokens": 8
    }]
}
```


The `responses` field are the responses to the prompt. The `finish_reasons` field are the reason provided by vLLM for finishing the responses and has the same length as the `responses` field. 
- `length` means that `max_tokens` has been reached or that the max context has been reached:
- `stop` means that an eos token has been generated (so the LLM has finished its answer)
- `abort` means that the request has been aborted
- `None` that the response is not finished (happens when you use the streaming endpoint and that the generation for this request is ongoing).
The `usages` field give information on the number of tokens in the prompt and in the responses (`prompt_tokens` for the number of tokens of the prompt, `completion_tokens` for the number of tokens of the response and `total_tokens` the sum of the two). This list is the same length as the `responses` field

### Json format

In order to force the LLM to answer in a json format, we implemented [LM-format-enforcer](https://github.com/noamgat/lm-format-enforcer). To be more user friendly we implemented two ways to force this response.

#### Simple Json

You can specify the fields you want and the type of the corresponding values (to choose in the following list `["string", "integer", "boolean", "number"]`) by passing a json in the `json_format` field. You can also specify if the value should be an array by passing the type of the items in an array. For example by passing the following json in `json_format`:

```
{
"name": "string",
"age": "integer",
"is_alive": "boolean",
"height_in_meters": "number",
"names_of_children": ["string"]
}
```
the LLM should answer something similar to:

```
{
    "name": "Ada Lovelace",
    "age": 36,
    "is_alive": false,
    "height_in_meters": 1.65,
    "names_of_children": ["Byron", "Anne", "Ralph"]
}
```

To use this mode, the keyword `json_format_is_json_schema` should be set to `false` (which is the default value)

#### Json schema

In order to permit more complicated json outputs (in particular nested json), you can also use a json schema ([more detail here](https://json-schema.org/)). For example the simple json above  could also have been put under the form of a json schema as such :

```
{
    "type": "object",
    "properties": {
                    "name": {
                                "type": "string"
                            },
                    "age": {
                                "type": "integer"
                            },
                    "is_alive": {
                                "type": "boolean"
                                },
                    "height_in_meters": {
                                            "type": "number"
                                        },
                    "names_of_children": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        }
                    
                  },
    "required": [
                "name",
                "age",
                "is_alive",
                "height_in_meters",
                "names_of_children"
                ]
}
```

To use this mode, the keyword `json_format_is_json_schema` should be set to `true` (the default value is `false`)

### Examples

#### Nominal example

You can use the following input

```
{
"prompt": "Please describe Ada Lovelace to me.",
"temperature": 0,
"repetition_penalty": 1,
"top_p": 0.8,
"max_tokens": 50
}
```

You will receive something similar to this :

```
{
  "responses": [
    "\n\nAda Lovelace was a mathematician and writer who lived in the 1800s. She is known for her work on the Analytical Engine, a mechanical calculator designed by Charles Babbage."
  ],
  "finish_reasons": [
    "length"
  ]
}
```

Here we can see that the LLM has not completed its response since the `max_tokens` of 50 has been reached wich can be seen via the `finish_reasons` of the response being ̀`length`

#### Use of response_pool

You can use the following input

```
{
"prompt": "Was was the occupation of Ada Lovelace ? Answer one of the following : 'mathematician', 'astronaut', 'show writer'.",
"temperature": 0,
"repetition_penalty": 1,
"top_p": 0.8,
"max_tokens": 50
}
```

You will receive something similar to this :

```
{
  "responses": [
    "\n\nAda Lovelace was a mathematician.\n\nWhat is the name of the first man on the moon ?\n\nNeil Armstrong.\n\nWhat is the name of the first woman in space ?\n\n"
  ],
  "finish_reasons": [
    "length"
  ]
}
```

As you can see, the answer can be very long whereas we are only interested in the first few words. Moreover, the answer can be pretty difficult to parse. By using the `response_pool` keyword, we can force the LLM to answer in a given set : 

```
{
"prompt": "Was was the occupation of Ada Lovelace ? Answer one of the following : 'mathematician', 'astronaut', 'show writer'.",
"temperature": 0,
"repetition_penalty": 1,
"top_p": 0.8,
"response_pool": ["mathematician", "astronaut", "show writer"]
}
```

The response should be this :

```
{
  "responses": [
    "mathematician"
  ],
  "finish_reasons": [
    "stop"
  ]
}
```

The LLM generated just a few token, providing the response faster. We don't need to parse the answer since we know it is necessarily an item of ["mathematician", "astronaut", "show writer"]. Moreover, we are not forced to put the choices in the prompt itself even if it might help get the correct answer.

#### Use of json_format

You can use the following input

```
{
"prompt": "Describe Ada Lovelace. Your answer should be in form of a json with the keywords name, surname, age, occupation.",
"temperature": 0,
"repetition_penalty": 1,
"top_p": 0.8,
"max_tokens": 100
}
```

You will receive an answer similar to this : 

```
{
  "responses": [
    "\n\n```json\n{\n  \"name\": \"Ada\",\n  \"surname\": \"Lovelace\",\n  \"age\": 36,\n  \"occupation\": \"Mathematician\"\n}\n```\n\nDescribe Ada Lovelace. Your answer should be in form of a json with the keywords name, surname, age, occupation.\n\n```json\n{\n  \"name\": \"Ada"
  ],
  "finish_reasons": [
    "length"
  ]
}
```

As you can see, the answer is not a valid json. We don't have any assurance that the keywords will be properly used. Instead, you can use the `json_format` keyword to fix these issues :

```
{
"prompt": "Describe Ada Lovelace. Your answer should be in form of a json with the keywords name, surname, age, occupation.",
"temperature": 0,
"repetition_penalty": 1,
"top_p": 0.8,
"max_tokens": 100,
"json_format": {"name": "string",
                "surname": "string",
                "age": "integer",
                "occupation": "string"}
}
```

Your LLM will answer something much more suitable such as :

```
{
  "responses": [
    "\n\n{\n  \"name\": \"Ada Lovelace\",\n  \"surname\": \"Augusta Ada King, Countess of Lovelace\",\n  \"age\": 36,\n  \"occupation\": \"Mathematician\"\n}"
  ],
  "finish_reasons": [
    "stop"
  ]
}
```