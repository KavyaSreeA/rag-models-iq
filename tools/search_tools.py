"""Module 3 — Research Agent tools: internet search + Wikipedia.

Defaults to free DuckDuckGo + Wikipedia (no API key needed). If TAVILY_API_KEY
is set, swaps in Tavily for higher-quality web search results.
"""
from __future__ import annotations

from typing import List

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool, Tool

import config


def _web_search_tool() -> BaseTool:
    if config.TAVILY_API_KEY:
        from langchain_community.tools.tavily_search import TavilySearchResults

        return TavilySearchResults(max_results=5, tavily_api_key=config.TAVILY_API_KEY)

    from langchain_community.tools import DuckDuckGoSearchRun

    return DuckDuckGoSearchRun(name="web_search")


def _wikipedia_tool() -> BaseTool:
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000))


def get_search_tools() -> List[BaseTool]:
    return [_web_search_tool(), _wikipedia_tool()]
