"""
Analyst Agent - Simulation result evaluation
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from kokoa.config import Config
from kokoa.state import AgentState


USER_PROMPT = """Goal: {goal}
Success: {is_success}
Conductivity: {conductivity} S/cm (target: ~1.97e-6 S/cm)
Error: {error_log}
"""

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Evaluate kMC simulation results.

{user_prompt}

DECIDE:
- FINISH: Goal met or max iterations
- RETRY: Bug or result far from target  
- ROLLBACK: Result worse than before

OUTPUT (JSON only):
{{
    "status": "FINISH" | "RETRY" | "ROLLBACK",
    "reason": "Brief analysis",
    "code_issues": ["issue1", "issue2"],
    "next_instruction": "What to fix"
}}"""),
])


def analyst_node(state: AgentState, llm) -> dict:  
    result = state.get("simulation_output")
    python_code = state.get("python_code", "")
    research_log = state.get("research_log", [])
    
    if result is None:
        return {
            "status": "RETRY",
            "user_feedback": "No simulation result.",
            "research_log": research_log + ["Analyst: No result"]
        }
    
    user_prompt_vars = {
        "goal": state["goal"],
        "is_success": result.is_success,
        "conductivity": result.conductivity if result.conductivity else "N/A",
        "error_log": (result.error_message or "None")[:500]
    }
    
    # Format user prompt for display and LLM
    formatted_user_prompt = USER_PROMPT.format(**user_prompt_vars)
    
    # Print USER PROMPT
    print("[Analyst] USER PROMPT:")
    print(formatted_user_prompt)
    
    # Stream LLM response
    print("[Analyst] RESPONSE:")
    full_response = ""
    prompt_with_context = ANALYST_PROMPT.format_messages(user_prompt=formatted_user_prompt)
    for chunk in llm.stream(prompt_with_context):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
        full_response += content
    print("\n")
    
    try:
        cleaned = full_response.replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(cleaned)
    except:
        evaluation = {"status": "RETRY", "reason": "Parse failed", "next_instruction": "Retry"}
    
    # Calculate error rate
    current_val = result.conductivity if result.conductivity else 0.0
    target_val = 1.97e-6  # S/cm
    error_rate = abs(target_val - current_val) / target_val * 100 if current_val else 100.0
    
    status = evaluation.get("status", "RETRY")
    
    if state.get("iteration_count", 0) >= Config.MAX_LOOPS:
        status = "FINISH"
    
    return {
        "status": status,
        "current_error_rate": error_rate,
        "user_feedback": evaluation.get("next_instruction", ""),
        "research_log": research_log + [f"Analyst: {status}"]
    }


def create_analyst_node(llm):
    def node_fn(state: AgentState) -> dict:
        return analyst_node(state, llm)
    return node_fn
