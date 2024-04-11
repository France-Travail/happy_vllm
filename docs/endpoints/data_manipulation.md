# Data manipulation endpoints

In this section we will give more details on the endpoints `/metadata_text` and `/split_text`.

## metadata_text (POST)

Returns the number of tokens of a text and indicates the part that would be truncated if too long. Note that this endpoint uses the special version of the tokenizer provided by happy_vLLM (more details [here](tokenizer.md#vanilla-tokenizer-vs-happy_vllm-tokenizer)). The format of the input is as follows:

```
{
  "text": "Hey, how are you ?",
  "truncation_side": "left",
  "max_length": 2
}
```

 - `text`: The text we want to analyze
 - `truncation_side`: The side of the truncation. This keyword is optional and the default value is the one of the tokenizer which can be obtained for example via the [`/info` endpoint](technical.md#info-get)
 - `max_length`: The maximal length of the string before the truncation acts. This keyword is optional and the default value is the `max_model_len` of the model which can be obtained for example via the [`/info` endpoint](technical.md#info-get)

The format of the output is as follows:

```
{
  "tokens_nb": 6,
  "truncated_text": "Hey, how are"
}
```

 - `tokens_nb`: The number of tokens in the given text
 - `truncated_text`: The part of the text which would be truncated

## split_text (POST)

Splits a text in chunks. You can specify a minimal number of tokens present in each chunk. Each chunk will be delimited by separators you can specify. 

The format of the input is as follows:

```
{
  "text": "Hey, how are you ? I am clearly fine. And you ? Exceptionally good, thanks for asking.",
  "num_tokens_in_chunk": 4,
  "separators": [".", "!", "?", "|", " .", " !", " ?", " |"]
}
```
 
 - `text`: The text to split
 - `num_tokens_in_chunk`: The **minimal** number of tokens you want in a chunk. The keyword is optional, the default value is 200.
 - `separators`: The list of separators which can separate the chunks. Note that they should be corresponding to tokens of the tokenizer. That's why it might be a good practice, depending on the specific tokenizer, to include a space in the separator (such as this : ` ?`). This keyword is optional. The default value is [".", "!", "?", "|", " .", " !", " ?", " |"]

The format of the output is as follows:

```
{
  "split_text": [
    "Hey, how are you ?",
    " I am clearly fine.",
    " And you ? Exceptionaly good, thanks for asking."
  ]
}
```

 - `split_text`: The list of chunks obtained by splitting the text. Note that in this example, in the last chunk, even though " ?" is a separator, it was not split in two since " And you ?" is less than 4 tokens.