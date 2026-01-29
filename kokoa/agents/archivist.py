"""
Archivist Agent - Batch Mode
============================
Generates technical reports for single-pass batch updates.
Analyzes Success/Failure of the holistic strategy.
"""

import difflib
import math
import os
import json
from datetime import datetime
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from kokoa.config import Config
from kokoa.state import AgentState
from kokoa.tools import web_search, format_search_results

# ============================================================
# Technical Report Prompt
# ============================================================
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a scientific analyst reviewing a kMC simulation update.

**TASK:**
Write a concise Technical Report. This will be stored in a vector database for future scientists to learn from.

**GUIDELINES:**
1. Classify as SUCCESS (error decreased), NEUTRAL (no change), or FAILURE (error/crash).
2. Explain WHY it succeeded or failed using physics principles.
3. Cite the provided references to support your analysis.
4. Provide actionable advice for future experiments.

**REPORT FORMAT:**
## Result: [SUCCESS/NEUTRAL/FAILURE]

## Analysis
(Why did it succeed or fail? What physics principles apply? Review the code changes briefly.)

## Advice for Future Scientists
(2-3 specific, actionable recommendations)
- If SUCCESS: What to try next?
- If FAILURE: What to avoid and what alternatives exist?
"""),
    ("user", """
**CONTEXT**
- Strategy: {strategy}
- Target Assumptions: {targets}
- Result: {result_type}

**DATA**
[Error Analysis]:
- Initial Log Error: {prev_log_error:+.2f} (vs experiment, positive = overestimate)
- Final Log Error: {curr_log_error:+.2f} (vs experiment, positive = overestimate)
- Improved: {improved} (|Final| < |Initial| means closer to experiment)
- Direction: {direction}

[Code Diff Summary]:
{diff_summary}

[Execution Result]:
- Success: {is_success}
- Target Log Conductivity: {target_log}
- Result Log Conductivity: {result_log}
- Error Message: {error_message}

[Scientist Analysis (Batch Details)]:
{batch_details_str}

Write the report.
""")
])

# ============================================================
# Search Query Generation Prompt
# ============================================================
SEARCH_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a researcher preparing to write a technical report on a kMC simulation update."),
    ("user", """
    Context:
    - Strategy: {strategy}
    - Relaxed Assumptions: {relaxed_hurdles_summary}
    - Outcome: {result_type} (Improved: {improved})

    What single Google search query would best help you find theoretical explanations or similar studies to discuss in your report? 
    Return ONLY the query string, no quotes or explanation.
    """)
])

# ============================================================
# Helper Functions
# ============================================================

def _calculate_log_error(conductivity: float, target: float = Config.TARGET_CONDUCTIVITY) -> float:
    """Calculate signed log error. Positive = overestimate, Negative = underestimate"""
    if conductivity is None or conductivity <= 0: return 10.0
    return math.log10(conductivity) - math.log10(target)  # No abs()

def _generate_diff(previous_code: str, current_code: str) -> str:
    if not previous_code: return "(First pass - comparing against baseline)"
    
    diff = difflib.unified_diff(
        previous_code.splitlines(keepends=True),
        current_code.splitlines(keepends=True),
        fromfile='previous.py',
        tofile='current.py',
        lineterm=''
    )
    diff_text = ''.join(diff)
    return diff_text if diff_text else "(No changes detected)"

def _classify_result(is_success: bool, error_delta: float) -> str:
    if not is_success: return "FAILURE"
    elif error_delta < 0: return "SUCCESS"
    else: return "NEUTRAL"

def _format_batch_details(relaxed_hurdles: List[dict]) -> str:
    parts = []
    for h in relaxed_hurdles:
        parts.append(f"### {h.get('name', 'Unknown')}")
        parts.append(f"**Reason:** {h.get('reason', '')}")
        parts.append(f"**Changes:** {h.get('changes', '')}")
        parts.append("")
    return "\n".join(parts)

# ============================================================
# Agent Node
# ============================================================

def archivist_node(state: AgentState, llm) -> dict:
    """Archivist: Batch Report Generation"""
    print("\n📜 [Archivist] Generating Batch Technical Report...")
    
    result = state.get("simulation_output")
    current_code = state.get("current_code", "")
    previous_code = state.get("previous_code", "")
    
    # Load previous code from file if missing (Baseline)
    if not previous_code:
        try:
            with open(os.path.join(Config.INITIAL_STATE_DIR, "initial_state.py"), "r", encoding="utf-8") as f:
                previous_code = f.read()
        except: previous_code = ""

    # Calculate Metrics
    target = Config.TARGET_CONDUCTIVITY
    curr_cond = result.conductivity if result and result.conductivity else 0.0
    
    # Calculate Log Values for Display
    if target > 0:
        target_log = f"{math.log10(target):.2f}"
    else:
        target_log = "Undefined"
        
    if curr_cond > 0:
        result_log = f"{math.log10(curr_cond):.2f}"
    else:
        result_log = "FAIL"
    
    # Baseline comparison (Always compare against Initial State in Single Pass)
    try:
        with open(os.path.join(Config.INITIAL_STATE_DIR, "initial_state.json"), "r") as f:
            init_data = json.load(f)
            init_cond = init_data.get("conductivity", 0.0)
            prev_log_error = _calculate_log_error(init_cond)
    except:
        prev_log_error = 10.0
        
    curr_log_error = _calculate_log_error(curr_cond)
    error_delta = curr_log_error - prev_log_error
    
    # Improvement check: |final| < |initial| means closer to experiment
    improved = "Yes" if abs(curr_log_error) < abs(prev_log_error) else "No"
    
    # Direction: overestimate (positive) or underestimate (negative)
    if curr_log_error > 0.1:
        direction = "Overestimate (simulation σ > experiment σ)"
    elif curr_log_error < -0.1:
        direction = "Underestimate (simulation σ < experiment σ)"
    else:
        direction = "On target (within 0.1 orders)"
    
    # Generate Diff
    diff_text = _generate_diff(previous_code, current_code)
    
    # Classify
    is_success = result.is_success if result else False
    result_type = _classify_result(is_success, error_delta)
    
    # Search References (Simplified for Batch: Search Strategy Keywords)
    relaxed_hurdles = state.get("relaxed_hurdles", [])
    target = state.get("target_assumption", "")
    
    strategy_str = "Batch Optimization"
    if relaxed_hurdles:
        strategy_str = relaxed_hurdles[0].get("name", "Batch Optimization")
    
    print(f"   Searching references for: {strategy_str}")
    references = "No references."
    try:
        # Generate dynamic search query
        search_query_chain = SEARCH_QUERY_PROMPT | llm | StrOutputParser()
        query = search_query_chain.invoke({
            "strategy": strategy_str,
            "relaxed_hurdles_summary": _format_batch_details(relaxed_hurdles),
            "result_type": result_type,
            "improved": improved
        }).strip().strip('"')
        
        print(f"   Generated Query: {query}")
        
        results = web_search(query, max_results=2)
        references = format_search_results(results)
    except Exception as e:
        print(f"   Search failed: {e}")

    # Generate Report
    try:
        chain = ANALYSIS_PROMPT | llm | StrOutputParser()
        
        batch_details_str = _format_batch_details(relaxed_hurdles)
        
        report = chain.invoke({
            "strategy": strategy_str,
            "targets": target,
            "result_type": result_type,
            "target_log": target_log,
            "result_log": result_log,
            "prev_log_error": prev_log_error,
            "curr_log_error": curr_log_error,
            "improved": improved,
            "direction": direction,
            "is_success": is_success,
            "error_message": result.error_message if result else "None",
            "diff_summary": diff_text,
            "batch_details_str": batch_details_str
        })
        
        print(f"   -> Report Generated ({len(report)} chars)")
        
        # Extract assumption ID from target (e.g., "A1", "A2")
        import re
        assumption_id_match = re.search(r'\b(A\d+)\b', target, re.IGNORECASE)
        target_assumption_id = assumption_id_match.group(1).upper() if assumption_id_match else ""
        
        # 1. Save to Vector Store (embedding-only: Analysis + Advice)
        from kokoa.memory import save_to_memory
        save_to_memory(
            content=report,  # Only the LLM-generated content, no diff
            collection="technical_reports",
            metadata={
                "run_id": state.get("run_id"),
                "type": "BATCH",
                "result_type": result_type,
                "target": target,
                "target_assumption_id": target_assumption_id  # For filtering
            },
            force=True
        )
        print(f"   -> Saved to technical_reports (embedding, assumption={target_assumption_id})")
        
        # 2. Save Full Report to Central Folder (technical_reports/)
        run_id = state.get("run_id", "unknown")
        central_reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "technical_reports")
        os.makedirs(central_reports_dir, exist_ok=True)
        
        # Include references in the report with AI generation markers
        full_report = f"""<!-- Generated by {Config.MODEL_NAME} -->

# Batch Technical Report

{report}

<!-- End of AI-generated content -->

## References
{references}

## Full Diff
```diff
{diff_text}
```"""
        report_filename = f"{run_id}_{result_type}.md"
        central_report_path = os.path.join(central_reports_dir, report_filename)
        with open(central_report_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"   -> Saved report to {central_report_path}")
        
    except Exception as e:
        print(f"   Report generation failed: {e}")

    return {
        "research_log": state["research_log"] + ["Archivist: Batch Report Complete"]
    }

def create_archivist_node(llm):
    def node_fn(state: AgentState) -> dict:
        return archivist_node(state, llm)
    return node_fn
