# Tokenizer

## Tokenizer endpoints

The tokenizer endpoints allow to use the tokenizer underlying the model. These endpoints are [`/v2/tokenizer`](#v2tokenizer-post) and [`/v2/decode`](#v2decode-post) and you can find more details on each below.

:warning: **Deprecated** These endpoints [`/v1/tokenizer`](#v1tokenizer-post) and [`/v1/decode`](#v1decode-post) are deprecated

### /v2/tokenizer (POST)
Tokenizes the given text. The format of the input is as follows according to the method:

#### Completions

```
{
  "model": "my_model",
  "prompt": "This is a text example",
  "add_special_tokens": true
}
```

  - `model`: ID of the model to use
  - `prompt` : The text to tokenize
  - `add_special_tokens` : Add a special tokens to the begin (optional, default value : `true`)

#### Chat/Completions

```
{
  "model": "my_model",
  "messages": [
    {
      "role": "system",
      "content": "This is an example"
    },
    {
      "role": "user",
      "content": "This is an example"
    }
  ],
  "add_special_tokens": true,
  "add_generation_prompt": true
}
```

 - `model` : ID of the model to use
 - `messages` : The texts to tokenize
 - `add_special_tokens` : Add a special tokens to the begin (optional, default value : `false`)
 - `add_generation_prompt` : Add generation prompt's model in decode response (optional, default value : `true`)

The format of the output is as follows :

```
{
  "count": [
    23
  ],
  "max_model_len": 8192,
  "tokens": [128000, 2028, 374, 264, 1495, 318]
}
```

 - `count`: The number of token in the input
 - `max_model_len`: Max model length in config
 - `tokens`: The list of token ids given by the tokenizer (give one extra token only if `add_special_tokens` was set to `true` in the request)


### /v2/decode (POST)

Decodes the given token ids. The format of the input is as follows :

```
{
  "tokens": [128000, 2028, 374, 264, 1495, 318],
  "model": "my_model"
}
```

 - `tokens`: The ids of the tokens we want to decode
 - `model`: ID of the model to use

The format of the output is as follows:

```
{
  "prompt": "<s> Hey, how are you ?"
}
```

 - `prompt`: The decoded string corresponding to the token ids


### /v1/tokenizer (POST) :warning: **Deprecated**

Tokenizes the given text. The format of the input is as follows :

```
{
  "text": "text to tokenizer",
  "with_tokens_str": true,
  "vanilla": true
}
```

 - `text`: The text to tokenize
 - `with_tokens_str`: Whether the list of tokens in string form should be given in the response (optional, default value : `false`)
 - `vanilla`: Whether we should use the vanilla version of the tokenizer or the happy_vLLM version (see [this section](#vanilla-tokenizer-vs-happy_vllm-tokenizer) for more details). This keyword is optional and the default value is `true`.

The format of the output is as follows :

```
{
  "tokens_ids": [
    1,
    17162,
    28725,
    910,
    460,
    368,
    1550
  ],
  "tokens_nb": 7,
  "tokens_str": [
    "<s>",
    "▁Hey",
    ",",
    "▁how",
    "▁are",
    "▁you",
    "▁?"
  ]
}
```

 - `tokens_ids`: The list of token ids given by the tokenizer
 - `tokens_nb`: The number of tokens in the input
 - `tokens_str`: The string representation of each token (given only if `with_tokens_str` was set to `true` in the request)

### /v1/decode (POST) :warning: **Deprecated**

Decodes the given token ids. The format of the input is as follows :

```
{
  "token_ids": [
    1,
    17162,
    28725,
    910,
    460,
    368,
    1550
  ],
  "with_tokens_str": true,
  "vanilla": true
}
```

 - `token_ids`: The ids of the tokens we want to decode
 - `with_tokens_str`: Whether we want the response to also decode the ids, id by id
 - `vanilla`: Whether we should use the vanilla version of the tokenizer or the happy_vLLM version (see [this section](#vanilla-tokenizer-vs-happy_vllm-tokenizer) for more details). This keyword is optional and the default value is `true`.

The format of the output is as follows:

```
{
  "decoded_string": "<s> Hey, how are you ?",
  "tokens_str": [
    "<s>",
    "▁Hey",
    ",",
    "▁how",
    "▁are",
    "▁you",
    "▁?"
  ]
}
```

 - `decoded_string`: The decoded string corresponding to the token ids
 - ̀`tokens_str`: The decoded string for each token id (given only if `with_tokens_str` was set to `true` in the request)

## Vanilla tokenizer vs happy_vLLM tokenizer

Using the endpoints `/v1/tokenizer` and `/v1/decode`, you can decide if you want to use the usual version of the tokenizers (with the keyword `vanilla` set to `true`). But in some cases, the tokenizer introduces special characters instead of whitespaces, adds a whitespace in front of the string etc. While it is usually the correct way to use the tokenizer (since the models have been trained with these), in some cases, you might want just to get rid of all these additions. We provide a simple way to do so just by setting the keyword `vanilla` to `false` in the endpoints `/v1/tokenizer` and `/v1/decode`.

For example, if you want to encode and decode the string : `Hey, how are you ? Fine thanks.` with the Llama tokenizer, it will create the following tokens (in string forms) : 

For the usual tokenizer:

`["<s>", "▁Hey", ",", "▁how", "▁are", "▁you", "▁?", "▁Fine", "▁thanks", "."]` 

For the happy_vLLM tokenizer:

`["H", "ey", ",", " how", " are", " you", " ?", " Fine", " thanks", "."]`

 Note that the "Hey" is not treated the same way, that the whitespaces are directly translated in real whitespaces and there is no initial whitespace.

Note that our modified version of the tokenizer is the one used in the `/v1/metadata_text` endpoint (see [this section](data_manipulation.md#metadata_text-post) for more details).