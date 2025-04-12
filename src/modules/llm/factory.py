from .providers import LLMProvider

class LLMFactory:
    """Factory to create LLM providers."""

    @staticmethod
    def create_provider(provider: str = "google") -> LLMProvider:
        return LLMProvider(provider)