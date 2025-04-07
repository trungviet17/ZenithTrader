from langgraph.graph import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage


class GeneralState(TypedDict):
    """
    General state for the agent.
    """

    messages : Annotated[list[str], add_messages]  