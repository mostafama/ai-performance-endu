"""
model_api.py
------------
Low-level functions for sending a prompt to each LLM provider and returning
a standardized 4-tuple: (response_text, elapsed_seconds, error_bool, error_message)

Each provider has its own SDK with slightly different calling conventions:
  - OpenAI / Groq / OpenRouter: chat completions API (same interface)
  - Gemini: GenerativeModel object with GenerationConfig
  - Anthropic: messages.create with content blocks

The top-level query_model() function dispatches to the right provider
based on the model's "api" field in config.MODELS_CONFIG.
"""

import time
import config


# ---------------------------------------------------------------------------
# Per-provider query functions
# Each returns: (text, elapsed_sec, error: bool, error_message: str)
# ---------------------------------------------------------------------------

def _query_openai(prompt, cfg, client):
    """Send a prompt via the OpenAI chat completions API."""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=cfg["model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return resp.choices[0].message.content or "", time.time() - t0, False, ""
    except Exception as exc:
        return "", time.time() - t0, True, str(exc)[:200]


def _query_gemini(prompt, cfg, genai_module):
    """Send a prompt via the Google Gemini GenerativeModel API."""
    t0 = time.time()
    try:
        model = genai_module.GenerativeModel(cfg["model_id"])
        resp = model.generate_content(
            prompt,
            generation_config=genai_module.GenerationConfig(
                temperature=cfg["temperature"],
                max_output_tokens=cfg["max_tokens"],
            ),
        )
        return resp.text, time.time() - t0, False, ""
    except Exception as exc:
        return "", time.time() - t0, True, str(exc)[:200]


def _query_anthropic(prompt, cfg, client):
    """Send a prompt via the Anthropic messages API (Claude models)."""
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=cfg["model_id"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks; we want the first text block
        text = resp.content[0].text if resp.content else ""
        return text, time.time() - t0, False, ""
    except Exception as exc:
        return "", time.time() - t0, True, str(exc)[:200]


def _query_groq(prompt, cfg, client):
    """Send a prompt via Groq's chat completions API (same shape as OpenAI)."""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=cfg["model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return resp.choices[0].message.content or "", time.time() - t0, False, ""
    except Exception as exc:
        return "", time.time() - t0, True, str(exc)[:200]


def _query_openrouter(prompt, cfg, client):
    """
    Send a prompt via OpenRouter.
    OpenRouter uses the OpenAI SDK format, so we reuse _query_groq's logic.
    """
    return _query_groq(prompt, cfg, client)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def query_model(model_name, prompt, clients):
    """
    Send a prompt to the specified model using the appropriate provider client.

    Args:
        model_name: key from config.MODELS_CONFIG (e.g. "claude-sonnet-4.5")
        prompt:     the full prompt string to send
        clients:    dict of initialized provider clients from clients.py

    Returns:
        (text, elapsed_sec, error: bool, error_message: str)
    """
    cfg = config.MODELS_CONFIG[model_name]
    api = cfg["api"]

    dispatch = {
        "openai":     ("openai",     _query_openai),
        "gemini":     ("gemini",     _query_gemini),
        "anthropic":  ("anthropic",  _query_anthropic),
        "groq":       ("groq",       _query_groq),
        "openrouter": ("openrouter", _query_openrouter),
    }

    if api not in dispatch:
        return "", 0.0, True, f"Unknown API type: {api}"

    client_key, fn = dispatch[api]
    client = clients.get(client_key)
    if not client:
        return "", 0.0, True, f"{api.capitalize()} client not available"

    return fn(prompt, cfg, client)
