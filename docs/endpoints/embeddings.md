# Embeddings

## Embeddings endpoints

### /v1/embeddings

Mirror of the `/v1/embeddings` endpoint of vLLM. Note that only this endpoint has been implemented in happy_vLLM (in particular `/pooling` and `/score` are not implemented). More details [here](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#embeddings-api). Note that happy_vLLM must be launched with the argument `--task embed` in order for this endpoint to be available.

The format of the input is as follows :

```
{
  "input": ["First text", "Seconde text"],
  "model": "my_model",
  "encoding_format": "float"
}
```

And the output is of the form:

```
{
  "id": "embd-418265ed36ab48e5bd153ac6e9d33c24",
  "object": "list",
  "created": 1736946607,
  "model": "my_model",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "embedding": [float_1, float_2, .....]
      },
      {
      "index": 1,
      "object": "embedding",
      "embedding": [float_1, float_2, .....]}
      ]
}
```