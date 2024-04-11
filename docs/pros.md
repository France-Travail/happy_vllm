# What are the bonuses of using happy_vLLM ?

happy_vLLM provides several functionalities useful for a production purpose. Here are a few.

## Environment variables

All the arguments used to launch the api and load the model can be specified via three methods : environment variables, .env files or cli arguments (see [this section](arguments.md) for more details). This way, you can easily specify the arguments according to your different environments : dev, pre-production, production, ...


## New endpoints

happy_vLLM add new endpoints useful for the users wich don't need to set up their own instances (for example for a tokenizer) but can directly use those provided by this API. For more details on endpoints, click [here](endpoints/endpoints.md)

If you would like to see an endpoint added, don't hesitate to open an issue or a PR.

## Already included logits processors, easy to use

happy_vLLM include some logits processors which provide new functionalities to the generation which can simply be accessed via keywords passed to the generation request. Namely:

 - The possibility to force the LLM to answer in a set of possible answer. Useful, for example, when you want to use the LLM as a classifier, you can force it to answer only in constrained way to be sure to always have a valid output without any parsing of the response.
 - The possibility to force the LLM to answer using a json making the parsing of the answer a piece of cake. The specification of this json are made as simple as possible in order to permit beginner users to use this functionality. 

More details on how to use these [here](endpoints/generate.md)

## Swagger

A well documented swagger (the UI being reachable at the `/docs` endpoint) in order for users not so used to using API to be able to quickly get the hang of it and be as autonomous as possible in querying the LLM. 