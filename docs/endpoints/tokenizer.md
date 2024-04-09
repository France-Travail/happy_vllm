# Tokenizer

## Tokenizer endpoints

The tokenizer endpoints allow to use the tokenizer underlying the model.

### tokenizer (POST)

Tokenizes the given text. The input is of the form :

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

The output is of the form :

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

### decode (POST)

Decodes the given token ids. The input is of the form :

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

The output is of the form:

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

Using the routes `tokenizer` and `decode`, you can decide if you want to use the usual version of the tokenizers (with the keyword `vanilla` set to `true`). But in some cases, the tokenizer introduce special characters instead of whitespaces, add a whitespace in front of the string etc. While it is usually the correct way to use the tokenizer (since the models have been trained with these), in particular cases, you might want to just get rid of all these additions. We provide a simple way to do so just by setting the keyword `vanilla` to `false` in the routes `tokenizer` and `decode`.

For example, if you want to encode and decode the string : `Hey, how are you ? Fine thanks.` with the Llama tokenizer, it will create the following tokens (in string forms) : `["<s>", "▁Hey", ",", "▁how", "▁are", "▁you", "▁?", "▁Fine", "▁thanks", "."]` for the usual tokenizer and `["H", "ey", ",", " how", " are", " you", " ?", " Fine", " thanks", "."]` for the modified. Note in particular that the "Hey" is not treated the same way but that the whitespaces are directly translated in real whitespaces and there is no initial whitespace.

Note that this modified version of the tokenizer is the one used in the `metadata_text` route (see [this section](data_manipulation.md#metadata_text-post) for more details) but otherwise, the usual tokenizer is used (in particular for the `generate` and `generate_stream` routes)