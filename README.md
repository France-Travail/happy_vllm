To install this library :
 - `pip install -e .` or `pip install happy-vllm`

How to launch : 

 - `python src/happy_vllm/launch.py` or via the entry point : `happy-vllm`

The path to the model is specified (in the order of priority):
 - In the environnement variable `MODEL`
 - In the `MODEL` field of the `.env` 
 - In the argument `--model` of `launch.py` (or of `happy-vllm`)
If no model are specified, the default model of vllm is chosen

The name of the model is specified (in the order of priority):
 - In the environnement variable `MODEL_NAME`
 - In the `MODEL_NAME` field of the `.env` 
 - In the argument `--model_name` of `launch.py` (or of `happy-vllm`)

 ### Routes
  - #### /generate (POST) :
  Generates a completion of a prompt. The body is of the form:
  ```
  {"prompt": "prompt_to_complete" (str),
      "prompt_in_response": if_we_want_the_prompt_in_the_response (bool),
      "response_pool": ["First_possible_response", "Second_possible_response", ...] (list of str),
      "min_tokens": min_number_of_tokens_generated (int),
      "json_format": a simple json or a json schema to force the generation of a json as response of the LLM,
      "json_format_is_json_schema": indicates if the json_format is a simple json or a json schema (bool),
      "all_keywords_from_vllm_except_logits_processors": (see https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py) }
  ```
  `response_pool` forces the model to answer in this list. Note that `response_pool` and `min_tokens` can't be both present. Same thing for `response_pool` and `json_format`.
  
  ##### How to use `json_format` and `json_format_is_json_schema`
  The logits processor json format forces the LLM to respond in a json with a format provided by the user. The format can be of two type : 
  - A flat json where the keys are strings and the values are the expected type in the following list `['string', 'integer', 'boolean', 'number']`. You can also put values equal to a list containing a single element in the previous list. The LLM will then understand that you expect an array containing elements of the indicated type. The argument `json_format_is_json_schema` must be set to `False` (which is the default value). For example, you can use a json of the following form:
    - ```{"age": "integer", "name": "string", "is_married": "boolean", "height_in_meter": "number", "diplomas": ["string"]}```
  - For more complex json format, use a valid json schema as described for example here `https://json-schema.org/learn/getting-started-step-by-step`. The argument `json_format_is_json_schema` must be set to `True`
  
  The output is of the form:
  ```
  {
  "responses": [
    "First response", "Second response", "Third response"...
  ],
  "finish_reasons": [
    "length", "stop", "None", ...
  ],
  "prompt": "The_prompt"
}
  ```
  `responses` is the list of responses with length corresponding to the parameter `n` in the input (see vllm github for more details)

  `finish_reasons` is a list of the same length as `responses` and indicates the reasons for the response to be finished. `"length"` means that `max_tokens` has been reached, `"stop"` that an eos token has been generated, `"abort"` that the request has been aborted,`"None"` that the response is not finished

  `prompt` gives back the prompt. It is present only if `prompt_in_response` was set to `True` in the input

  - #### /generate_stream (POST) :
  Same as the route `/generate` but with stream

  - #### /split_text (POST) : 
  Splits a text in chunks containing at least a given number of tokens and ending by a separator :
  ```
  {"text": "Text to split",
      "num_tokens_in_chunk": 10,
      "separators":[".", "!", "?", "|", " .", " !", " ?", " |"]}
  ```
  Note that the `separators` field is optional (the default value is [".", "!", "?", "|", " .", " !", " ?", " |"]). Also note that the `separators` used are the tokenization of the strings and not the strings themselves (which explains why we must for example specify ' .' and '.' as two separate separators) 
  The output will be of the form: 
   ```
  {"split_text": ["My", "split", "text"] (list of str)
     }
  ```

  - #### /tokenizer (POST) : 
  Tokenizes a text in a list of tokens_ids. We can also get back the list of tokens in str form with the corresponding keyword `with_tokens_str`. You can also use the keyword `vanilla` to specify if you want to use the usual tokenizer or the modified version (explained here : link to the explanation)
  ```
  {"text": "Text to tokenize" (str),
      "with_tokens_str": true (bool, default value : false),
      "vanilla": true (bool, default value: true)
     }
  ```
  The output will be of the form:
   ```
  {"tokens_ids": [ids_of_the_tokens] (list of int)
      "tokens_nb": nb_of_tokens (int),
      "tokens_str": [tokens_in_str_form] (list of str)
     }
  ```
  The field `tokens_str` is only present if `with_tokens_str` is `True`

  - #### /decode (POST) : 
  Decodes a list of token ids. You can also use the keyword `vanilla` to specify if you want to use the usual tokenizer or the modified version (explained here : link to the explanation)
  ```
  {"token_ids": token_ids (list of int),
      "with_tokens_str": true (bool, default value : false),
      "vanilla": true (bool, default value: true)
     }
  ```
  The output will be of the form:
   ```
  {"decoded_string": "the decoded string" (string)
      "tokens_nb": nb_of_tokens (int),
      "tokens_str": [tokens_in_str_form] (list of str)
     }
  ```
  The field `tokens_str` is only present if `with_tokens_str` is `True`

  - #### /metadata_text (POST) : 
  Gives the number of tokens in a text and the truncated part. This uses the modified version of the tokenizer as described here : link_to_doc
  ```
  {"text": "Text to analyze",
      "truncation_side": "right",
      "max_length": 2048
     }
  ```
  Note that `truncation_side` and `max_length` are optional and the default values are those of the underlying model
  The output is of the form:
  ```
  {"nb_tokens": The total number of tokens (int),
      "truncated_text": "the_truncated_text" (str)
     }
  ```

  - #### /info (get) :
  Outputs :
  ```
  {
  "application": "happy_vllm",
  "version": "version of the library" (str),
  "model_name": "The name of the model" (str),
  "truncation_side": "The truncation side of the tokenizer" (str),
  "max_length": the_max_length_for_prompt_plus_generation (int)
}
  ```

  - #### /live_metrics (GET) :
  Outputs:
  ```
  {
  "requests_running": number_of_requests_running (int),
  "requests_swapped": number_of_requests_swapped (int),
  "requests_pending": number_of_requests_pending (int),
  "gpu_cache_usage": percentage_of_gpu_cache_usage (float),
  "cpu_cache_usage": percentage_of_cpu_cache_usage (float)
}
  ``` 


### Tokenizers

Using the routes `tokenizer` and `decode`, you can decide if you want to use the usual version of the tokenizers (with the keyword `vanilla` set to `true`). But in some cases, the tokenizer introduce special character instead of whitespaces, add a whitespace in front of the string etc. While it is usually the correct way to use the tokenizer (since the models have been trained with these), in particular cases, you might want to just get rid of all these additions. We provide a simple way to do so just by setting the keyword `vanilla` to `false` in the routes `tokenizer` and `decode`.

For example, if you want to encode and decode the string : `Hey, how are you ? Fine thanks.` with the Llama tokenizer, it will create the following tokens (in string forms) : `["<s>", "▁Hey", ",", "▁how", "▁are", "▁you", "▁?", "▁Fine", "▁thanks", "."]` for the usual tokenizer and `["H", "ey", ",", " how", " are", " you", " ?", " Fine", " thanks", "."]` for the modified. Note in particular that the "Hey" is not treated the same way but that the whitespaces are directly translated in real whitespaces and there is no initial whitespace.

Note that this modified version of the tokenizer is the one used in the `metadata_text` route but otherwise, the usual tokenizer is used (in particular for the `generate` and `generate_stream` routes)