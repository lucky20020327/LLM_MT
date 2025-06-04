from openai import OpenAI

def call_LLM(prompt, api_key, baseLLM="deepseek"):
    if baseLLM == "deepseek":
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        model_name = "deepseek-chat"
    else:
        raise NotImplementedError(
            f"Base LLM {baseLLM} is not implemented. Please use 'deepseek'."
        )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()