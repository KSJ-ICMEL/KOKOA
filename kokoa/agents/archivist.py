"""
Archivist Agent - LLM-Powered Knowledge Archiving Specialist
============================================================
- 기술 보고서 생성 (Diff, Result, Discussion)
- Tavily 검색으로 레퍼런스 수집
- 로그 스케일 오차율 계산
"""

import difflib
import math
from datetime import datetime
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from kokoa.config import Config
from kokoa.state import AgentState


# ============================================================
# Technical Report Prompt
# ============================================================
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a scientific analyst reviewing kMC simulation experiments.

Analyze the experiment and write a technical report in the following format:

## Result: [SUCCESS/NEUTRAL/FAILURE]

**Criteria:**
- SUCCESS: Hypothesis properly implemented AND log error decreased
- NEUTRAL: Hypothesis properly implemented BUT log error increased  
- FAILURE: Code error prevented conductivity calculation

## Code Changes
(Summarize the diff in 2-3 sentences - what physics was added/modified?)

## Discussion
(Analyze WHY this result occurred. Use the provided references if relevant, or explain based on physics principles. 300 tokens max.)

References (if provided):
{references}
"""),
    ("user", """
[Iteration]: {iteration}
[Hypothesis]: {hypothesis}

[Code Diff]:
```diff
{diff}
```

[Execution Result]:
- Success: {is_success}
- Conductivity: {conductivity} S/cm
- Previous Log Error: {prev_log_error:.2f} orders
- Current Log Error: {curr_log_error:.2f} orders
- Error Delta: {error_delta:+.2f} orders

[Error Message (if any)]:
{error_message}

Write the technical report:
""")
])


def _calculate_log_error(conductivity: float, target: float = Config.TARGET_CONDUCTIVITY) -> float:
    """Calculate log-scale error (orders of magnitude)"""
    if conductivity is None or conductivity <= 0:
        return float('inf')
    return abs(math.log10(conductivity) - math.log10(target))


def _generate_diff(previous_code: str, current_code: str) -> str:
    """Generate unified diff between two code versions"""
    if not previous_code:
        return "(First iteration - no previous code)"
    
    diff = difflib.unified_diff(
        previous_code.splitlines(keepends=True),
        current_code.splitlines(keepends=True),
        fromfile='previous.py',
        tofile='current.py',
        lineterm=''
    )
    diff_text = ''.join(diff)
    
    # Truncate if too long
    if len(diff_text) > 2000:
        diff_text = diff_text[:2000] + "\n... (truncated)"
    
    return diff_text if diff_text else "(No changes detected)"


def _search_references(hypothesis: str, result_type: str) -> str:
    """Search Tavily for relevant references"""
    try:
        from tavily import TavilyClient
        import os
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "(No Tavily API key - using prior knowledge)"
        
        client = TavilyClient(api_key=api_key)
        
        # Generate search query based on result
        if result_type == "SUCCESS":
            query = f"kMC simulation {hypothesis} mechanism physical explanation"
        elif result_type == "NEUTRAL":
            query = f"kMC simulation error increase despite improvement {hypothesis}"
        else:
            query = f"kMC simulation code error {hypothesis}"
        
        response = client.search(query=query, max_results=3)
        
        refs = []
        for i, result in enumerate(response.get("results", [])[:3]):
            refs.append(f"[{i+1}] {result.get('title', 'N/A')}\n    {result.get('content', '')[:200]}")
        
        return "\n\n".join(refs) if refs else "(No relevant references found)"
        
    except Exception as e:
        return f"(Search failed: {e})"


def _classify_result(is_success: bool, error_delta: float) -> str:
    """Classify result as SUCCESS/NEUTRAL/FAILURE"""
    if not is_success:
        return "FAILURE"
    elif error_delta < 0:  # Error decreased
        return "SUCCESS"
    else:  # Error increased or same
        return "NEUTRAL"


def archivist_node(state: AgentState, llm) -> dict:
    """Archivist: Generate technical report and archive knowledge"""
    print("[Archivist] Analyzing experiment and generating technical report...")
    
    result = state.get("simulation_output")
    python_code = state.get("python_code", "")
    previous_code = state.get("previous_code", "")
    hypothesis = state.get("hypothesis", "")
    run_dir = state.get("run_dir")
    iteration = state.get("iteration_count", 0)
    research_log = state.get("research_log", [])
    prev_error_rate = state.get("current_error_rate", 100.0)
    
    target = Config.TARGET_CONDUCTIVITY
    
    # Calculate log-scale errors
    curr_conductivity = result.conductivity if result and result.conductivity else 0.0
    prev_log_error = math.log10(prev_error_rate / 100 * target + target) - math.log10(target) if prev_error_rate < 100 else 10.0
    curr_log_error = _calculate_log_error(curr_conductivity) if curr_conductivity > 0 else 10.0
    error_delta = curr_log_error - prev_log_error
    
    # Generate diff
    diff_text = _generate_diff(previous_code, python_code)
    print(f"   [1/4] Generated diff ({len(diff_text)} chars)")
    
    # Classify result
    is_success = result.is_success if result else False
    result_type = _classify_result(is_success, error_delta)
    print(f"   [2/4] Result classification: {result_type}")
    
    # Search references via Tavily
    references = _search_references(hypothesis, result_type)
    print(f"   [3/4] Searched references")
    
    # Generate technical report via LLM
    try:
        chain = ANALYSIS_PROMPT | llm | StrOutputParser()
        
        report = chain.invoke({
            "iteration": iteration,
            "hypothesis": hypothesis[:500] if hypothesis else "N/A",
            "diff": diff_text,
            "is_success": is_success,
            "conductivity": curr_conductivity,
            "prev_log_error": prev_log_error,
            "curr_log_error": curr_log_error,
            "error_delta": error_delta,
            "error_message": result.error_message if result and result.error_message else "None",
            "references": references
        })
        print(f"   [4/4] Generated technical report ({len(report)} chars)")
        
    except Exception as e:
        report = f"(Report generation failed: {e})"
        print(f"   [4/4] Report generation failed: {e}")
    
    # Save technical report to memory
    from kokoa.memory import save_to_memory
    
    full_report = f"""
# Technical Report - Iteration {iteration}

{report}

---
## Appendix: Code Diff
```diff
{diff_text}
```

Timestamp: {datetime.now().isoformat()}
""".strip()
    
    saved = save_to_memory(
        content=full_report,
        collection="technical_reports",
        metadata={
            "iteration": iteration,
            "result_type": result_type,
            "log_error": curr_log_error,
            "error_delta": error_delta,
            "conductivity": curr_conductivity
        },
        run_dir=run_dir
    )
    
    if saved:
        print(f"   -> Saved technical report to 'technical_reports'")
    
    return {
        "research_log": research_log + [f"Archivist: {result_type} - Report saved"],
        "previous_code": python_code  # Update previous_code for next iteration
    }


def create_archivist_node(llm):
    """Factory function to create archivist node with LLM"""
    def node_fn(state: AgentState) -> dict:
        return archivist_node(state, llm)
    return node_fn
