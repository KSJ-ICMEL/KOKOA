"""
Engineer Agent - kMC simulation code generation
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from kokoa.state import AgentState


USER_PROMPT = """Hypothesis: {hypothesis}

Code ({code_len} bytes):
{current_code}

Error: {last_error}

Implement the hypothesis. Use target_time = {simulation_time}.
"""

ENGINEER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a kMC simulation expert for solid electrolytes.

RULES:
1. NO 'kokoa' imports. Code must be standalone.
2. target_time = {simulation_time} seconds (fixed).
3. Use _CIF_PATH variable for CIF file (pre-injected).
4. Supercell: structure.make_supercell([3, 3, 3]) (fixed 3×3×3).
5. Print: print(f"Conductivity: {{val}} S/cm")
6. No subprocess/pip calls.

LIBRARIES: numpy, scipy, pymatgen, matplotlib

REQUIRED STRUCTURE:
- Load structure from _CIF_PATH
- KMCSimulator class with run_step(), calculate_properties()
- MSD → D → conductivity pipeline
- Track particle positions for MSD
- Print progress every 2000 steps: Step, Time(ns), MSD, sigma(mS/cm)

Output: Python code block only (```python ... ```). No explanations."""),
    ("user", USER_PROMPT)
])


def engineer_node(state: AgentState, llm) -> dict:
    current_code = state.get("python_code", "") or "# No existing code."
    
    last_error = "None"
    if state.get("simulation_output") and not state["simulation_output"].is_success:
        last_error = state["simulation_output"].error_message or "Unknown error"
    
    from kokoa.config import Config
    simulation_time = Config.SIMULATION_TIME
    
    # Truncate code for prompt
    code_preview = current_code[:1500] if len(current_code) > 1500 else current_code
    
    user_prompt_vars = {
        "hypothesis": state["hypothesis"],
        "current_code": code_preview,
        "code_len": len(current_code),
        "last_error": last_error[:500] if last_error else "None",
        "simulation_time": simulation_time
    }
    
    # Print USER PROMPT
    print("[Engineer] USER PROMPT:")
    print(USER_PROMPT.format(**user_prompt_vars))
    
    # Stream LLM response
    print("[Engineer] RESPONSE:")
    new_code_raw = ""
    for chunk in llm.stream(ENGINEER_PROMPT.format_messages(**user_prompt_vars)):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
        new_code_raw += content
    print("\n")
    
    cleaned_code = new_code_raw.replace("```python", "").replace("```", "").strip()
       
    return {
        "python_code": cleaned_code,
        "last_valid_code": state.get("python_code", ""),  # Save for rollback
        "research_log": state.get("research_log", []) + ["Engineer: Code updated."]
    }


def create_engineer_node(llm):
    def node_fn(state: AgentState) -> dict:
        return engineer_node(state, llm)
    return node_fn
