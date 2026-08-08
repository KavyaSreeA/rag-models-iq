"""Module 13 — Python tool: calculations, table formatting, CSV processing.

Wraps LangChain's PythonAstREPLTool with pandas pre-imported so the agent can
do things like "calculate employee attendance percentage" or "create a
summary table" by writing and executing short Python snippets.
"""
from __future__ import annotations

import pandas as pd
from langchain_core.tools import BaseTool
from langchain_experimental.tools.python.tool import PythonAstREPLTool


def get_python_tool() -> BaseTool:
    tool = PythonAstREPLTool(locals={"pd": pd})
    tool.description = (
        "Executes Python code for calculations, table formatting, report generation, "
        "and CSV/pandas processing. `pd` (pandas) is pre-imported. Input must be a "
        "single valid Python snippet; the last expression's value (or an explicit "
        "print()) is returned as the result. Example: "
        "'attendance_pct = (18/20)*100; attendance_pct'"
    )
    return tool
