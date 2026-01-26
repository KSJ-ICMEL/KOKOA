"""
Archivist Agent - LLM-Powered Knowledge Archiving Specialist
============================================================
- 기술 보고서 생성 (Diff, Result, Discussion)
- Tavily 검색으로 레퍼런스 수집
- 로그 스케일 오차율 계산
"""

import difflib
import math
import os
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

Your task is to analyze the experiment and write a technical report.

Classification criteria:
- SUCCESS: Hypothesis correctly implemented AND log error decreased
- NEUTRAL: Hypothesis correctly implemented BUT log error increased  
- FAILURE: Code error prevented conductivity calculation

**Write the report in exactly this format:**

## Result: [SUCCESS/NEUTRAL/FAILURE]

## Code Changes
(Copy the whole diff from [Code Diff] section here.)

## Discussion
(Analyze WHY this result occurred. Reference the provided sources if relevant, or explain based on physics principles.)

References (if provided):
{references}
"""),
    ("user", """
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

[Code Agent Fix Report]:
{debug_summary}

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
            refs.append(f"[{i+1}] {result.get('title', 'N/A')}\n    {result.get('content', '')}")
        
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
    scientist_code = state.get("scientist_code", python_code) # Use Scientist's code for diff (intended logic)
    previous_code = state.get("previous_code", "")
    
    # Fallback: If no previous code (1st iteration), use initial_state.py
    if not previous_code:
        try:
            with open(os.path.join(Config.INITIAL_STATE_DIR, "initial_state.py"), "r", encoding="utf-8") as f:
                previous_code = f.read()
            print("   [Diff] Using initial_state.py as baseline")
        except Exception:
            previous_code = ""

    hypothesis = state.get("hypothesis", "")
    run_dir = state.get("run_dir")
    iteration = state.get("iteration_count", 0)
    research_log = state.get("research_log", [])
    # prev_log_error is retrieved directly from state (default 10.0)
    prev_log_error = state.get("current_log_error", 10.0)
    
    target = Config.TARGET_CONDUCTIVITY
    
    # Calculate log-scale errors
    curr_conductivity = result.conductivity if result and result.conductivity else 0.0
    
    # For first iteration, use initial_state.json conductivity as baseline
    if iteration <= 1:
        try:
            import json
            initial_json_path = os.path.join(Config.INITIAL_STATE_DIR, "initial_state.json")
            with open(initial_json_path, "r", encoding="utf-8") as f:
                initial_data = json.load(f)
            initial_conductivity = initial_data.get("conductivity", 0.0)
            prev_log_error = _calculate_log_error(initial_conductivity) if initial_conductivity > 0 else 10.0
            print(f"   [Baseline] Using initial_state.json conductivity: {initial_conductivity} S/cm")
        except Exception:
            prev_log_error = 10.0
    
    curr_log_error = _calculate_log_error(curr_conductivity) if curr_conductivity > 0 else 10.0
    error_delta = curr_log_error - prev_log_error
    
    # Generate diff (Compare previous vs Scientist's intended code)
    diff_text = _generate_diff(previous_code, scientist_code)
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
            "hypothesis": hypothesis if hypothesis else "N/A",
            "diff": diff_text,
            "is_success": is_success,
            "conductivity": curr_conductivity,
            "prev_log_error": prev_log_error,
            "curr_log_error": curr_log_error,
            "error_delta": error_delta,
            "error_message": result.error_message if result and result.error_message else "None",
            "debug_summary": state.get("debug_summary", "No debugging needed."),
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
        "previous_code": python_code,  # Update previous_code for next iteration
        "current_log_error": curr_log_error  # Store log error directly
    }


def create_archivist_node(llm):
    """Factory function to create archivist node with LLM"""
    def node_fn(state: AgentState) -> dict:
        return archivist_node(state, llm)
    return node_fn
