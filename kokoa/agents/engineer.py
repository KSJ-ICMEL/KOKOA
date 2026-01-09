"""
Engineer Agent
==============
가설을 기반으로 시뮬레이션 코드 작성
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from kokoa.state import AgentState


ENGINEER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Python coding expert specializing in kinetic Monte Carlo (kMC) simulations for solid electrolytes.
Your task: Implement the Theorist's hypothesis by writing or modifying Python code.

[CRITICAL CONSTRAINTS]
1. DO NOT modify `target_time`. Simulation time is fixed at 10 ns (competition deadline).
2. MUST print conductivity in exact format: print(f"Conductivity: {{val}} S/cm")
3. NO `pip install` or `subprocess` calls. Use only pre-installed libraries.

[AVAILABLE LIBRARIES]
Standard: numpy, scipy, matplotlib, pandas
Materials Science: pymatgen, ase, matminer, diffpy
kMC Specific: If kmos or similar is available, feel free to use it.
You may also try other common scientific Python libraries - if unavailable, the error will be caught and you can retry.

[ENCOURAGED MODIFICATIONS - Be Creative!]
You are encouraged to explore novel physics implementations:
- Novel hop rate formulations (angle-dependent, local environment-dependent)
- Concentration-dependent activation energies
- Multi-ion correlation effects (beyond exclusion principle)
- Alternative energy barrier calculations (NEB-inspired, bond-order)
- Advanced neighbor selection (Voronoi, coordination shell)
- Temperature-dependent prefactors
- Defect clustering effects
- Strain-dependent conductivity

[General Guidelines]
1. If creating a plot, save as 'result.png'.
2. Handle errors robustly (try-except).
3. Maintain MSD calculation → Diffusion coefficient → Conductivity pipeline.
4. Preserve particle tracking for proper MSD computation.

[CODE STYLE - Token Efficiency]
- Write concise, minimal code. Avoid verbose comments.
- Use short but meaningful variable names (e.g., src, tgt, vec, msd).
- English only. No emojis, no non-ASCII characters.
- Minimize print statements. Only essential outputs.
- Inline simple operations. Avoid unnecessary intermediate variables.

[Output Format]
Provide ONLY the Python code block (```python ... ```). No explanations.
"""),
    ("user", """
[Current Hypothesis]
{hypothesis}

[Existing Code]
{current_code}

[Instruction]
Update the code to implement the hypothesis. 
If previous run failed (Error: {last_error}), fix the bug.
""")
])


def engineer_node(state: AgentState, llm) -> dict:
    print("🔧 [Engineer] 시뮬레이션 코드 작성 중...")
    
    current_code = state.get("python_code", "") or "# No existing code."
    
    last_error = "None"
    if state.get("simulation_output") and not state["simulation_output"].is_success:
        last_error = state["simulation_output"].error_message or "Unknown error"
    
    chain = ENGINEER_PROMPT | llm | StrOutputParser()
    new_code_raw = chain.invoke({
        "hypothesis": state["hypothesis"],
        "current_code": current_code,
        "last_error": last_error
    })
    
    cleaned_code = new_code_raw.replace("```python", "").replace("```", "").strip()
    
    print(f"   → 코드 작성 완료 ({len(cleaned_code)} bytes)")
    
    return {
        "python_code": cleaned_code,
        "research_log": state.get("research_log", []) + ["Engineer: Code updated."]
    }


def create_engineer_node(llm):
    def node_fn(state: AgentState) -> dict:
        return engineer_node(state, llm)
    return node_fn
