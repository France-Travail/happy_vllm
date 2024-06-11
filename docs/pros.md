# What are the bonuses of using happy_vLLM ?

happy_vLLM provides several functionalities useful for a production purpose. Here are a few.

## Environment variables

All the arguments used to launch the api and load the model can be specified via three methods : environment variables, .env files or cli arguments (see [this section](arguments.md) for more details). This way, you can easily specify the arguments according to your different environments : dev, pre-production, production, ...


## New endpoints

happy_vLLM add new endpoints useful for the users wich don't need to set up their own instances (for example for a tokenizer) but can directly use those provided by this API. For more details on endpoints, click [here](endpoints/endpoints.md)

If you would like to see an endpoint added, don't hesitate to open an issue or a PR.

## Swagger

A well documented swagger (the UI being reachable at the `/docs` endpoint) in order for users not so used to using API to be able to quickly get the hang of it and be as autonomous as possible in querying the LLM. 

## Benchmarks

We developped a library [benchmark_llm_serving](https://github.com/France-Travail/benchmark_llm_serving) which provides a more complete benchmark of the vLLM serving API than the vanilla one.