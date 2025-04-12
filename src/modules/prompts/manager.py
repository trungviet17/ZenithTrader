from .templates import PromptTemplate
from typing import Dict, Any

class PromptManager:
    """Manage prompt templates for various tasks."""

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}

    def load_template(self, name: str):
        """Load a prompt template."""
        self.templates[name] = PromptTemplate(name)

    def get_prompt(self, name: str, **kwargs) -> str:
        """Get formatted prompt."""
        if name not in self.templates:
            self.load_template(name)
        return self.templates[name].format(**kwargs)