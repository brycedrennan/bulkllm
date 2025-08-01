# Adding a new model

Adding a new model means it should show up in three places: litellm models, llm_configs.py, and rate_limits.py

 - *litellm* Generally running the scraper, `make update-models` will pull in the latest models from major LLM providers, which will automatically populate the litellm models.
 - *llm_configs.py* add a LLMConfig object
        slug: dash separated slug ending with release date YYYYMMDD
        display_name: the text name displayed on the frontend,
        company_name=: company that created the model
        litellm_model_name: the litellm model name
        llm_family: similar to the litellm modle name but without any date or version. used to group different versions of the model
        temperature: default_temperature,
        max_tokens: default_max_tokens,
        thinking_config={},
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        release_date=date(2025, 4, 28),
 - *rate_limits.py* 
  

