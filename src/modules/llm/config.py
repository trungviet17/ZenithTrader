from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMConfig:
    """Configuration for LLM providers."""

    def __init__(self):
        self.providers = {
            "google": {
                "model": "gemini-1.5-pro",
                "api_key": os.getenv("GOOGLE_API_KEY"),
                "temperature": 0.7
            }
        }

    def get_config(self, provider: str) -> Dict[str, Any]:
        return self.providers.get(provider, {})