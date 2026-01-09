"""
Analyst Agent
=============
시뮬레이션 결과 평가 및 다음 액션 결정
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from kokoa.config import Config
from kokoa.state import AgentState


ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a rigorous Scientific Reviewer.
Evaluate the simulation code and results against the research goal.

[Goal]
{goal}

[Python Code]
```python
{python_code}
```

[Simulation Results]
- Success: {is_success}
- Conductivity: {conductivity}
- Error Log: {error_log}
- Execution Log (truncated): {execution_log}

[Task]
1. If simulation failed:
   - Analyze the error log AND the code to determine cause.
   - Identify if it's a Syntax error, Logic error, Missing library, or Physics issue.
   
2. If successful:
   - Compare conductivity with target (~1e-3 S/cm if not specified).
   - Evaluate if the physics in the code is reasonable.
   
3. Decide next step:
   - "FINISH": Goal met or max iterations reached.
   - "RETRY": Code bug or result far from target.
   - "ROLLBACK": Result significantly worse than before.

Output JSON only (no markdown):
{{
    "status": "FINISH" | "RETRY" | "ROLLBACK",
    "reason": "Detailed analysis of code and results...",
    "code_issues": ["List of specific issues found in code, if any"],
    "next_instruction": "Specific feedback for Theorist/Engineer on what to fix."
}}
"""),
])


def analyst_node(state: AgentState, llm) -> dict:
    print("📊 [Analyst] 결과 분석 중...")
    
    result = state.get("simulation_output")
    python_code = state.get("python_code", "")
    research_log = state.get("research_log", [])
    
    if result is None:
        return {
            "status": "RETRY",
            "user_feedback": "No simulation result available. Please run simulation first.",
            "research_log": research_log + ["Analyst: No result to analyze"]
        }
    
    code_preview = python_code[:3000] if len(python_code) > 3000 else python_code
    
    chain = ANALYST_PROMPT | llm | JsonOutputParser()
    
    try:
        evaluation = chain.invoke({
            "goal": state["goal"],
            "python_code": code_preview or "(initial_state.py - 코드 미제공)",
            "is_success": result.is_success,
            "conductivity": result.conductivity if result.conductivity else "N/A",
            "error_log": result.error_message or "None",
            "execution_log": result.execution_log[:1500]
        })
    except Exception as e:
        evaluation = {
            "status": "RETRY",
            "reason": f"Analysis failed: {e}",
            "code_issues": [],
            "next_instruction": "Retry generation."
        }
    
    current_val = result.conductivity if result.conductivity else 0.0
    target_val = 1.0e-3
    error_rate = abs(target_val - current_val) / target_val * 100 if current_val else 100.0
    
    status = evaluation.get("status", "RETRY")
    
    if state.get("iteration_count", 0) >= Config.MAX_LOOPS:
        status = "FINISH"
        evaluation["reason"] = evaluation.get("reason", "") + " (Max iterations reached)"
    
    code_issues = evaluation.get("code_issues", [])
    if code_issues:
        print(f"   ⚠️ 코드 이슈: {code_issues[:2]}...")
    
    print(f"   → 평가: {status} | {evaluation.get('reason', '')[:50]}...")
    
    return {
        "status": status,
        "current_error_rate": error_rate,
        "user_feedback": evaluation.get("next_instruction", ""),
        "research_log": research_log + [f"Analyst: {status} - {evaluation.get('reason', '')[:50]}"]
    }


def create_analyst_node(llm):
    def node_fn(state: AgentState) -> dict:
        return analyst_node(state, llm)
    return node_fn
