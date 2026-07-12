import os
import json
import time
import logging

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "sarvam": "sarvam-m",
}

DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "sarvam": "https://api.sarvam.ai/v1",
}


def get_client(provider, base_url=None):
    if provider == "anthropic":
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set in environment.")
        return Anthropic(api_key=api_key)

    if provider == "openai":
        from openai import OpenAI
        if base_url:
            api_key = os.environ.get("OPENAI_API_KEY", "not-needed-for-local-server")
            return OpenAI(api_key=api_key, base_url=base_url)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set in environment.")
        return OpenAI(api_key=api_key)

    if provider == "deepseek":
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY not set in environment.")
        return OpenAI(api_key=api_key, base_url=base_url or DEFAULT_BASE_URLS["deepseek"])

    if provider == "sarvam":
        from openai import OpenAI
        api_key = os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise SystemExit("SARVAM_API_KEY not set in environment.")
        return OpenAI(api_key=api_key, base_url=base_url or DEFAULT_BASE_URLS["sarvam"])

    raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic', 'openai', 'deepseek', or 'sarvam'.")


def call_llm_json(client, provider, prompt, model=None, max_tokens=1000, retries=3):
    model = model or DEFAULT_MODELS[provider]

    for attempt in range(retries):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
            elif provider in ("openai", "deepseek", "sarvam"):
                response = client.chat.completions.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.choices[0].message.content
            else:
                raise ValueError(f"Unknown provider: {provider!r}")

            if not raw:
                raise ValueError("empty response")
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)

        except Exception as e:
            logging.warning(f"LLM call failed (provider={provider}, attempt={attempt}): {e}")
            time.sleep(2 ** attempt)

    return None
