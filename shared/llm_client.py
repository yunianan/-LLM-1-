import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "..."),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


def ask_llm(prompt, model="deepseek-chat", temperature=0):
    """统一 LLM 调用边界：10 秒超时，最多进行 1 次 SDK 重试。"""
    bounded_client = client.with_options(timeout=10.0, max_retries=1)
    response = bounded_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = ask_llm("你好，请介绍一下你自己")
    print(result)
