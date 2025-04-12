from langchain_core.prompts import ChatPromptTemplate
from .config import PromptConfig

class PromptTemplate:
    """Manage prompt templates."""

    def __init__(self, name: str):
        config = PromptConfig().get_template(name)
        self.template = ChatPromptTemplate.from_template(config["template"])

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)