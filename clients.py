"""
clients.py
----------
Handles initializing API client objects for each LLM provider.

Each provider (OpenAI, Gemini, Anthropic, Groq, OpenRouter) needs its own
SDK and API key. This module checks which providers are actually needed for
a given run, then initializes only those clients — skipping any whose key
is missing or whose package isn't installed.

Returns a dict like: {"openai": <client>, "anthropic": <client>, ...}
"""

import os
import config


def init_clients(models):
    """
    Initialize API clients for all providers needed by the given model list.

    Args:
        models: list of model name strings (keys from config.MODELS_CONFIG)

    Returns:
        dict mapping provider name -> initialized client object
    """
    # Determine which API providers are actually needed for this run
    needed_apis = {config.MODELS_CONFIG[m]["api"] for m in models}
    clients = {}

    # --- OpenAI (used for GPT models) ---
    if "openai" in needed_apis:
        try:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY", "")
            if key:
                clients["openai"] = OpenAI(api_key=key)
            else:
                print("  OPENAI_API_KEY not set — OpenAI models will be skipped")
        except ImportError:
            print("  openai package not installed")

    # --- Google Gemini ---
    if "gemini" in needed_apis:
        try:
            import google.generativeai as genai
            key = os.getenv("GOOGLE_API_KEY", "")
            if key:
                # Gemini uses a module-level configure call rather than a client object
                genai.configure(api_key=key)
                clients["gemini"] = genai
            else:
                print("  GOOGLE_API_KEY not set — Gemini models will be skipped")
        except ImportError:
            print("  google-generativeai package not installed")

    # --- Anthropic (Claude models) ---
    if "anthropic" in needed_apis:
        try:
            from anthropic import Anthropic
            key = os.getenv("ANTHROPIC_API_KEY", "")
            if key:
                clients["anthropic"] = Anthropic(api_key=key)
            else:
                print("  ANTHROPIC_API_KEY not set — Anthropic models will be skipped")
        except ImportError:
            print("  anthropic package not installed")

    # --- Groq (used for LLaMA models via Groq's fast inference API) ---
    if "groq" in needed_apis:
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY", "")
            if key:
                clients["groq"] = Groq(api_key=key)
            else:
                print("  GROQ_API_KEY not set — Groq models will be skipped")
        except ImportError:
            print("  groq package not installed")

    # --- OpenRouter (used for DeepSeek and other models via unified gateway) ---
    # OpenRouter uses the OpenAI SDK but with a different base URL and API key
    if "openrouter" in needed_apis:
        try:
            from openai import OpenAI
            key = os.getenv("OPENROUTER_API_KEY", "")
            if key:
                # Optional headers identify your app to OpenRouter
                extra = {}
                site = os.getenv("OPENROUTER_SITE_URL", "")
                name = os.getenv("OPENROUTER_APP_NAME", "EduQuestion3")
                if site:
                    extra["HTTP-Referer"] = site
                if name:
                    extra["X-Title"] = name
                clients["openrouter"] = OpenAI(
                    api_key=key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers=extra if extra else None,
                )
            else:
                print("  OPENROUTER_API_KEY not set — OpenRouter models will be skipped")
        except ImportError:
            print("  openai package not installed")

    return clients
