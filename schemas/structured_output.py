"""Pydantic models used for LangChain structured-output parsing.

Module 12 (Structured Outputs) — every agent that produces a final report
returns a `TaskReport`. Module 9 (Conditional Routing) — the Coordinator
Agent classifies each request into a `RouteDecision`.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    HR = "hr"
    RESEARCH = "research"
    EMAIL = "email"
    DOCUMENT = "document"
    PYTHON = "python"
    MEMORY = "memory"           # remember/recall facts about the employee (Module 11)
    SEQUENTIAL = "sequential"   # multi-step chained workflow (Module 7)
    PARALLEL = "parallel"       # multi-entity parallel workflow (Module 8)


class RouteDecision(BaseModel):
    """Output of the Coordinator Agent's classifier (Modules 6 & 9)."""

    agent: AgentName = Field(description="Which agent/workflow should handle this request.")
    reasoning: str = Field(description="One short sentence on why this agent was chosen.")
    entities: List[str] = Field(
        default_factory=list,
        description=(
            "For 'research'/'parallel': the distinct subjects/companies mentioned "
            "(e.g. ['Google', 'Microsoft']). Empty list otherwise."
        ),
    )
    memory_action: Optional[str] = Field(
        default=None,
        description="Only for agent='memory': 'remember' or 'recall'.",
    )
    memory_key: Optional[str] = Field(
        default=None,
        description="Only for memory_action='remember': the profile field, e.g. 'department', 'name', 'email_style'.",
    )
    memory_value: Optional[str] = Field(
        default=None,
        description="Only for memory_action='remember': the value to store, e.g. 'Finance'.",
    )


class Priority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TaskReport(BaseModel):
    """Structured report every agent produces for the 'Generated Report' panel."""

    task_summary: str = Field(description="One to three sentence summary of what was done.")
    department: str = Field(description="The department this task relates to (HR, IT, Finance, Operations, General, ...).")
    priority: Priority = Field(default=Priority.MEDIUM, description="Urgency of the task.")
    actions_taken: List[str] = Field(default_factory=list, description="Concrete actions already performed.")
    pending_actions: List[str] = Field(default_factory=list, description="Actions still outstanding, if any.")
    recommended_next_steps: List[str] = Field(default_factory=list, description="Suggested next steps for the employee.")


class UserProfile(BaseModel):
    """Long-term memory record (Module 11)."""

    user_id: str
    name: Optional[str] = None
    department: Optional[str] = None
    preferred_email_style: Optional[str] = None
    faq_topics: List[str] = Field(default_factory=list)
    frequent_docs: List[str] = Field(default_factory=list)
