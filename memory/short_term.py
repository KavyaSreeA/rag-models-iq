"""Module 11 — short-term memory: the current conversation, in-process only.

Backed by st.session_state in the Streamlit app; this class is the plain-
Python core so it's independently testable without importing Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class ShortTermMemory:
    max_turns: int = 20
    messages: List[BaseMessage] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))
        self._trim()

    def as_list(self) -> List[BaseMessage]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
