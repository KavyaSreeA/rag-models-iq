"""Module 8 — Parallel Processing (RunnableParallel):

Executes multiple independent research tasks simultaneously (one per named
entity/company), then combines the results into a single comparison report.
Triggered by the Coordinator for requests like
"Research Google, Microsoft, Amazon, and OpenAI."
"""
from __future__ import annotations

from typing import List

from langchain_core.runnables import RunnableLambda, RunnableParallel

from agents.base import AgentResult, generate_report, get_llm
from agents.research_agent import run_research_agent


def _research_one(entity: str) -> str:
    result = run_research_agent(f"Research {entity} and summarize recent, relevant information.")
    return result.answer


def run_parallel_research(entities: List[str], original_request: str) -> AgentResult:
    entities = [e for e in entities if e.strip()] or [original_request]

    # Each branch runs independently; RunnableParallel invokes all branches
    # concurrently (via a thread pool under the hood) and waits for all of
    # them before returning the combined dict.
    parallel_map = RunnableParallel(
        **{f"entity_{i}": RunnableLambda(lambda _x, e=e: _research_one(e)) for i, e in enumerate(entities)}
    )
    raw_results = parallel_map.invoke({})
    findings = {entities[i]: raw_results[f"entity_{i}"] for i in range(len(entities))}

    llm = get_llm(temperature=0.2)
    comparison_prompt = (
        "Combine the following independent research findings into a single, "
        "well-organized comparison report (use headers per entity, then a brief "
        "'Comparison' section at the end):\n\n"
        + "\n\n".join(f"## {name}\n{text}" for name, text in findings.items())
    )
    answer = llm.invoke(comparison_prompt).content

    report = generate_report(
        "parallel",
        original_request,
        answer,
        [{"name": "research", "args": {"entity": e}} for e in entities],
    )
    result = AgentResult(agent="parallel", answer=answer, report=report)
    result.retrieved_docs = [{"entity": name, "excerpt": text[:500]} for name, text in findings.items()]
    return result
