"""
config.py - Swappable model configuration and global hyperparameters.
Supports swapping LLM providers (Gemini, OpenAI, Heuristic fallback) via env vars or config.
"""

import os

class Config:
    # Model configuration
    # Options: "heuristic" (local), "openai", or "compatible" (custom base URL).
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "heuristic")
    
    # Default model matches the default OpenAI base URL; MODEL_NAME can override it.
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1")
    MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
    MODEL_TEMPERATURE = 0.0
    PREFILTER_PER_EPISODE = int(os.getenv("PREFILTER_PER_EPISODE", "3"))
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Strict duration limits (enforced by code)
    MIN_PROMO_DURATION = float(os.getenv("MIN_PROMO_DURATION", "20.0"))
    MAX_PROMO_DURATION = float(os.getenv("MAX_PROMO_DURATION", "60.0"))
    
    # Max candidates evaluated per episode by LLM
    MAX_CANDIDATES_PER_EP = int(os.getenv("MAX_CANDIDATES_PER_EP", "10"))
    
    # Target top ranked promos across show
    TOP_RANKED_COUNT = int(os.getenv("TOP_RANKED_COUNT", "15"))
    
    # Confidentiality safeguard (redact raw screenplay dialogue for public deployment)
    CONFIDENTIAL_MODE = os.getenv("KUKU_CONFIDENTIAL_MODE", "0") == "1"

    @classmethod
    def update_model(cls, provider: str, model_name: str, api_key: str = ""):
        cls.LLM_PROVIDER = provider
        cls.MODEL_NAME = model_name
        if provider == "gemini" and api_key:
            cls.GEMINI_API_KEY = api_key
        elif provider == "openai" and api_key:
            cls.OPENAI_API_KEY = api_key
        elif provider == "compatible" and api_key:
            cls.MODEL_API_KEY = api_key
