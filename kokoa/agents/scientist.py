"""
Scientist Agent - Analysis & Implementation Architecture (Refactored)
=====================================================================
1. Analysis: Diagnoses missing physics and generates search queries.
2. Retrieval: Searches for necessary knowledge based on analysis.
3. Implementation: Synthesizes knowledge to generate the final code.
"""

import re
import math
import json
import os
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from kokoa.config import Config
from kokoa.state import AgentState
from kokoa.tools import web_search, extract_code_from_url
from kokoa.memory import search_memory


# ===============================================================================
# 1. ANALYSIS & QUERY PROMPT (Phase 1: Researcher)
# ===============================================================================

class ScientistAnalysis(BaseModel):
    diagnosis: str = Field(description="Brief explanation of why the current model is failing.")
    missing_physics: List[str] = Field(description="List of physical concepts to add (e.g., 'Coulomb interaction', 'Polaron hopping')")
    search_queries: List[str] = Field(description="Detailed search queries to find the missing info")

analysis_parser = PydanticOutputParser(pydantic_object=ScientistAnalysis)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Lead Computational Scientist analyzing a kMC simulation to improve its accuracy.

**OBJECTIVE:**
You must relax ONE SPECIFIC ASSUMPTION provided in [Target Assumption]. Focus ONLY on this assumption.
Do NOT diagnose unrelated issues. Your job is to find what physics/parameters are missing due to THIS assumption.

**TASK:**
1. Read the [Target Assumption] carefully - this is the ONLY thing you are relaxing.
2. Diagnose how this specific assumption causes the error (ignore other assumptions).
3. Identify the missing physics DIRECTLY related to this assumption.
4. Generate search queries to find formulas/parameters for relaxing THIS assumption.

{format_instructions}
"""),
    ("user", """[Goal]: {goal}

[Target Assumption to Relax]:
{target_assumption_detail}

[Current Status]:
- Target: {target} S/cm
- Current: {current_conductivity} S/cm
- Log Error: {log_error}

[Current Code (Read Only)]:
```python
{current_code}
```

Analyze how the [Target Assumption] causes the error and generate search queries to find the specific physics needed to relax it.""")
])

# ===============================================================================

# 1.5. REPORT QUERY PROMPT (For Technical Report RAG)
# ===============================================================================

REPORT_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You generate search queries to find relevant technical reports from past LLZO kMC simulation experiments.

Output ONLY a brief search query (5-10 words). The query should:
1. Use GENERAL physics concepts (e.g., "energy landscape", "activation barrier", "site occupancy") instead of specific values.
2. Focus on the CATEGORY of modification (e.g., "lattice relaxation", "site energy", "hopping rate") not implementation details.
3. Omit specific library names (pymatgen, numpy) unless the modification is about fixing library usage.

Examples:
- Good: "site energy differentiation kMC conductivity improvement"
- Bad: "24d 0.15eV 96h 0.0eV tetrahedral octahedral pymatgen"
- Good: "lattice relaxation migration barrier elastic strain"
- Bad: "E_A1 = 0.15 eV oxygen bottleneck 2.4 Angstrom" """),
    ("user", """I am about to modify: {modification}
Background: {diagnosis}

Generate a general search query to find past experiments with similar physics modifications.""")
])

# ===============================================================================

# 2. IMPLEMENTATION PROMPT (Phase 2: Architect & Developer)

# ===============================================================================

IMPLEMENTATION_PROMPT = ChatPromptTemplate.from_messages([
("system", """You are a Computational Scientist implementing improvements to a kMC simulation.

**GOAL:**
Synthesize the provided [Knowledge Context] to upgrade the [Current Code] and achieve the target accuracy.

**GUIDELINES:**
* **Evidence-Based:** Do not invent physics. Use the formulas found in [Knowledge Context].
* **Focused Change:** Implement only the improvements identified in the diagnosis. Do not add unrelated changes.
* **Learn from History:** Avoid approaches listed in [Past Failures]. Prefer approaches in [Proven Strategies].

**STRICT FORMATTING RULES:**
* Output ONLY the Python code block.
* No JSON, no markdown text outside the code block.
* You MUST start the code with the specific header provided below."""),
("user", """[Diagnosis of Current Failure]:
{diagnosis}

[Knowledge Context]:
{knowledge_context}

[Past Failures to Avoid]:
{past_failures}

[Proven Strategies]:
{proven_strategies}

[Current Code Base]:

```python
{current_code}
```

**CRITICAL CONSTRAINT:**
You MUST start your code EXACTLY with this block. Do not change a single character:

```python
import os, sys, json
import numpy as np
from pymatgen.core import Structure

# === 1. Structure Loading ===
script_dir = os.path.dirname(os.path.abspath(__file__))
cif_path = os.path.join(script_dir, "LLZO.cif")
structure = Structure.from_file(cif_path)
N = 4  # Supercell expansion
structure.make_supercell([N, N, N])
print(f"Supercell: {{N}}x{{N}}x{{N}}, Total atoms: {{len(structure)}}")
```

Generate the full improved Python script now.""")
])

# ===============================================================================

# HELPER FUNCTIONS

# ===============================================================================

def _search_knowledge(queries: List[str], run_dir: str, knowledge_retriever) -> str:
    """Execute search queries and compile context"""
    context_parts = []

    
    # RAG Search (Use first query or summary)
    primary_query = queries[0] if queries else "kMC simulation optimization"
    print(f"   [1/3] Searching RAG: {primary_query}...")
    try:
        retrieved_docs = knowledge_retriever.invoke(primary_query)
        if retrieved_docs:
            rag_text = "\n".join([f"[Paper] {d.page_content}" for d in retrieved_docs[:3]])
            context_parts.append(f"### Internal Papers\n{rag_text}")
    except Exception as e:
        print(f"     -> RAG Error: {e}")

    # Web Search (Loop through specific queries)
    print(f"   [2/3] Web Search ({len(queries)} queries)...")
    try:
        seen_urls = set()
        for q in queries[:3]: # Limit to top 3 queries
            results = web_search(f"{q} python code", max_results=2)
            for r in results:
                if r['url'] not in seen_urls:
                    seen_urls.add(r['url'])
                    context_parts.append(f"### Web: {r['title']}\n{r['snippet']}\nSource: {r['url']}")
                    
                    # Opportunistic code extraction
                    if "github" in r['url'] or "code" in r['url']:
                        codes = extract_code_from_url(r['url'])
                        if codes:
                            context_parts.append(f"### Extracted Code ({r['url']})\n```python\n{codes[0][:1000]}\n```")
    except Exception as e:
        print(f"     -> Web Search Error: {e}")

    return "\n\n".join(context_parts)

    

def _retrieve_reports(query: str, assumption_id: str = None, k: int = 3) -> dict:
    """Retrieve relevant technical reports, optionally filtered by assumption ID"""
    
    def fmt(docs, empty_msg):
        if not docs: return empty_msg
        return "\n".join([f"- {d.get('content','')} (Type: {d.get('metadata',{}).get('result_type')})" for d in docs])

    try:
        if not query or not query.strip():
            return {"failures": "No query provided", "successes": "No query provided"}

        # Build filter with assumption ID if provided
        failure_filter = {"result_type": {"$in": ["FAILURE", "NEUTRAL"]}}
        success_filter = {"result_type": "SUCCESS"}
        
        if assumption_id:
            failure_filter["target_assumption_id"] = assumption_id
            success_filter["target_assumption_id"] = assumption_id

        failures = search_memory(query + " failure problem", "technical_reports", k=k, include_global=True, filter=failure_filter)
        successes = search_memory(query + " success improvement", "technical_reports", k=k, include_global=True, filter=success_filter)
        
        # Fallback message indicates assumption-specific search
        no_failure_msg = f"No failure cases for {assumption_id} yet" if assumption_id else "No failure cases found yet"
        no_success_msg = f"No success cases for {assumption_id} yet" if assumption_id else "No success cases found yet"
        
        return {
            "failures": fmt(failures, no_failure_msg),
            "successes": fmt(successes, no_success_msg)
        }

    except Exception as e:
        print(f"     -> Report Search Error: {e}")
        return {"failures": "Search error", "successes": "Search error"}


# ===============================================================================
# AGENT NODE
# ===============================================================================

def scientist_node(state: AgentState, knowledge_retriever, llm) -> dict:
    """Scientist: Analysis -> Retrieval -> Implementation (Refactored)"""
    iteration = state.get("iteration_count", 0) + 1
    run_dir = state.get("run_dir")
    current_code = state.get("current_code", "")
    
    # Fallback: load initial_state.py if current_code is empty
    if not current_code:
        try:
            initial_state_path = os.path.join(Config.INITIAL_STATE_DIR, "initial_state.py")
            with open(initial_state_path, "r", encoding="utf-8") as f:
                current_code = f.read()
        except Exception as e:
            print(f"   ⚠️ Could not load initial_state.py: {e}")
    
    print(f"\n👨‍🔬 [Scientist] Analysis & Implementation")

    # 0. Print Status Table
    sim_output = state.get("simulation_output")
    target_cond = Config.TARGET_CONDUCTIVITY
    target_log_cond = math.log10(target_cond)
    
    # Load Init
    init_cond, init_log_cond = 0.0, 0.0
    try:
        with open(os.path.join(Config.INITIAL_STATE_DIR, "initial_state.json"), "r") as f:
            d = json.load(f)
            init_cond = d.get("conductivity", 0.0)
            init_log_cond = math.log10(init_cond) if init_cond > 0 else 0.0  # Calculate directly
    except: pass
    
    # Formatted Print
    w_state, w_sigma, w_log, w_error = 10, 18, 18, 18
    header_fmt = f"| {{:^{w_state}}} | {{:^{w_sigma}}} | {{:^{w_log}}} | {{:^{w_error}}} |"
    row_fmt    = f"| {{:^{w_state}}} | {{:^{w_sigma}}} | {{:^{w_log}}} | {{:^{w_error}}} |"
    sep = "+" + "-"*(w_state+2) + "+" + "-"*(w_sigma+2) + "+" + "-"*(w_log+2) + "+" + "-"*(w_error+2) + "+"
    
    init_error = init_log_cond - target_log_cond  # Signed: + = overestimate
    
    print("\n" + sep)
    print(header_fmt.format("State", "σ_ion [S/cm]", "log(σ_ion)", "error (±)"))
    print(sep)
    print(row_fmt.format("Target", f"{target_cond:.5e}", f"{target_log_cond:.8f}", "-"))
    print(row_fmt.format("Initial", f"{init_cond:.5e}", f"{init_log_cond:.8f}", f"{init_error:+.8f}"))
    print(sep + "\n")

    # 1. ANALYSIS PHASE (Researcher)
    print("   🧠 Step 1: Analyzing Missing Physics...")
    
    current_conductivity = sim_output.conductivity if sim_output else 0.0
    log_error = 10.0  # Default (failure case)
    if current_conductivity > 0:
        log_error = math.log10(current_conductivity) - math.log10(target_cond)  # Signed: + = overestimate

    # Using the NEW Analysis Prompt and Parser
    # Extract target assumption from goal (e.g., "Relax A1" -> "A1")
    goal_text = state["goal"]
    assumption_id_match = re.search(r'\b(A\d+)\b', goal_text, re.IGNORECASE)
    target_assumption_detail = "No specific assumption targeted."
    
    if assumption_id_match:
        assumption_id = assumption_id_match.group(1).upper()
        for a in state["assumptions_checklist"]:
            if a["id"] == assumption_id:
                target_assumption_detail = f"[{a['id']}] {a['name']} ({a['category']})\nDescription: {a['description']}"
                break
    else:
        assumption_id = None
    
    analysis_chain = ANALYSIS_PROMPT.partial(
        format_instructions=analysis_parser.get_format_instructions()
    ) | llm | analysis_parser
    
    analysis_result: ScientistAnalysis = analysis_chain.invoke({
        "goal": state["goal"],
        "target_assumption_detail": target_assumption_detail,
        "target": f"{target_cond:.2e}",
        "current_conductivity": f"{current_conductivity:.2e}",
        "log_error": f"{log_error:.5f}",
        "current_code": current_code
    })
    
    print(f"   -> Diagnosis: {analysis_result.diagnosis}")
    print(f"   -> Missing Physics: {analysis_result.missing_physics}")
    print(f"   -> Queries: {analysis_result.search_queries}")

    # 2. RETRIEVAL PHASE
    print("   🔍 Step 2: Retrieving Knowledge...")
    knowledge_context = _search_knowledge(analysis_result.search_queries, run_dir, knowledge_retriever)
    
    # Search for reports based on the missing physics keywords + assumption ID filter
    reports_query = " ".join(analysis_result.missing_physics)
    reports = _retrieve_reports(reports_query, assumption_id=assumption_id)

    # 3. IMPLEMENTATION PHASE (Architect & Developer)
    print("   💻 Step 3: Implementing Improvements...")
    
    # Using the NEW Implementation Prompt
    coding_chain = IMPLEMENTATION_PROMPT | llm | StrOutputParser()
    
    response_content = coding_chain.invoke({
        "diagnosis": analysis_result.diagnosis,
        "knowledge_context": knowledge_context,
        "past_failures": reports["failures"],
        "proven_strategies": reports["successes"],
        "current_code": current_code
    })
    
    # Code Extraction
    python_code = ""
    code_match = re.search(r"```[ \t]*(?:python)?\s*\n(.*?)```", response_content, re.DOTALL | re.IGNORECASE)
    if code_match:
        python_code = code_match.group(1).strip()
    else:
        # Fallback for plain text response or malformed blocks
        if "import" in response_content and "structure" in response_content:
             python_code = response_content.strip()
        else:
             print("   ⚠️ Failed to extract code block. Falling back to previous code.")
             python_code = current_code 

    print(f"   -> Generated {len(python_code)} bytes of code")

    # Update State - use first missing_physics as single target
    target = analysis_result.missing_physics[0] if analysis_result.missing_physics else "Unknown"
    batch_summary = {
        "id": "BATCH_UPDATE",
        "name": f"Added: {target}",
        "reason": analysis_result.diagnosis[:500],
        "changes": target
    }

    return {
        "status": "CONTINUE",
        "hypothesis": f"## Diagnosis: {analysis_result.diagnosis}\n**Implemented:** {target}",
        "current_code": python_code,
        "iteration_count": iteration,
        "relaxed_hurdles": [batch_summary], 
        "target_assumption": assumption_id if assumption_id else target,  # Store assumption ID (e.g., "A1")
        "research_log": state["research_log"] + [f"Scientist: Implemented {target}"]
    }

def create_scientist_node(knowledge_retriever, llm):
    def node_fn(state: AgentState) -> dict:
        return scientist_node(state, knowledge_retriever, llm)
    return node_fn