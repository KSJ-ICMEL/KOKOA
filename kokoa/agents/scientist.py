"""
Scientist Agent - Computational Materials Scientist + Researcher
================================================================
Theory development + External knowledge search + Assumption review
- Reviews and relaxes idealized kMC assumptions
- Internal knowledge (RAG + Memory) search
- External knowledge (Tavily + arXiv) search
- Generates improved simulation code
"""

import re
import json
from typing import Optional, List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from kokoa.config import Config
from kokoa.state import AgentState
from kokoa.tools import web_search, extract_code_from_url
from kokoa.assumptions_config import (
    format_assumptions_for_prompt,
    relax_assumption,
    get_active_assumptions,
)


ASSUMPTION_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are analyzing kMC simulation assumptions to determine which should be relaxed.

**SIMULATION STATUS:**
- Target: 1.97e-6 S/cm (experimental LLZO conductivity)
- Current: {current_conductivity} S/cm
- Error: {error_rate}%

**TASK:**
1. Analyze why the current simulation deviates from reality
2. Identify which assumption most urgently needs to be relaxed
3. Explain the physical reality and how to implement the fix
4. If you discover a NEW gap not in the checklist, describe it as DISCOVERED_GAP

**OUTPUT FORMAT (JSON only):**
{{
    "analysis": "Brief analysis of error source",
    "selected_assumption_id": "A1-A10 or DISCOVERED",
    "reason_to_relax": "Why this assumption should be relaxed",
    "physical_reality": "What is the real physics",
    "implementation_plan": "How to implement in kMC code",
    "expected_improvement": "Expected effect on conductivity",
    "discovered_gap": "(Optional) New gap not in checklist"
}}"""),
    ("user", """{assumptions_checklist}

[Previous Error Message]:
{error_message}

[External Knowledge]:
{knowledge_context}

Select ONE assumption to relax and explain. Output JSON only.""")
])


SCIENTIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a **Computational Materials Scientist** specializing in kinetic Monte Carlo (kMC) simulations for solid-state electrolytes.

**YOUR ROLE:**
You are RELAXING the following assumption to make the simulation more realistic:
{assumption_being_relaxed}

**Implementation Plan:**
{implementation_plan}

**CODE REQUIREMENTS:**
1. NO 'kokoa' imports - code must be standalone
2. Use: numpy, scipy, pymatgen, matplotlib
3. Use `_CIF_PATH` variable for CIF file (injected at runtime)
4. target_time = {simulation_time} seconds
5. Supercell: structure.make_supercell([3, 3, 3])
6. Print: `print(f"Conductivity: {{val}} S/cm")`
7. Print progress every 2000 steps

**IMPORTANT:**
- Implement the assumption relaxation as specified
- Make ONE focused improvement
- The CodeAgent will fix any Python bugs later

**OUTPUT FORMAT (JSON only):**
{{
    "hypothesis": {{
        "title": "[A#] One-line improvement title",
        "mechanism": "Scientific explanation of assumption relaxation",
        "key_changes": ["change1", "change2"]
    }},
    "python_code": "```python\\nimport numpy as np\\n...complete code here...\\n```"
}}"""),
    ("user", """[Goal]: {goal}

[Current Simulation Code]:
```python
{current_code}
```

[Previous Result]:
{simulation_result}

[Knowledge Context]:
{knowledge_context}

Generate improved kMC simulation code that relaxes the target assumption. Output JSON only.""")
])


def _search_knowledge(query: str, run_dir: str, knowledge_retriever) -> str:
    """Search all knowledge sources and compile context"""
    from kokoa.memory import search_memory
    
    context_parts = []
    
    print("   [1/4] Searching RAG (papers)...")
    try:
        retrieved_docs = knowledge_retriever.invoke(query)
        if retrieved_docs:
            rag_text = "\n".join([
                f"[Paper {i+1}] {doc.page_content[:600]}" 
                for i, doc in enumerate(retrieved_docs[:3])
            ])
            context_parts.append(f"[Academic Papers]\n{rag_text}")
            print(f"         Found {len(retrieved_docs)} paper chunks")
    except Exception as e:
        print(f"         RAG failed: {e}")
    
    print("   [2/4] Searching Memory...")
    try:
        exp_results = search_memory(query, "experiments", k=3, run_dir=run_dir)
        insight_results = search_memory(query, "insights", k=2, run_dir=run_dir)
        
        if exp_results:
            memory_text = "\n".join([f"- {r['content'][:300]}" for r in exp_results])
            context_parts.append(f"[Past Experiments]\n{memory_text}")
            print(f"         Found {len(exp_results)} experiments")
        
        if insight_results:
            insight_text = "\n".join([f"- {r['content'][:200]}" for r in insight_results])
            context_parts.append(f"[Lessons Learned]\n{insight_text}")
            print(f"         Found {len(insight_results)} insights")
    except Exception as e:
        print(f"         Memory search failed: {e}")
    
    print("   [3/4] Web Search (Tavily)...")
    try:
        web_results = web_search(f"{query} python kMC simulation code", max_results=3)
        if web_results:
            web_text = "\n".join([
                f"[{r['title']}]\n{r['snippet'][:300]}" 
                for r in web_results[:3]
            ])
            context_parts.append(f"[Web Search Results]\n{web_text}")
            print(f"         Found {len(web_results)} web results")
            
            for r in web_results[:2]:
                url = r.get("url", "")
                if "github.com" in url or "example" in url.lower():
                    codes = extract_code_from_url(url)
                    if codes:
                        context_parts.append(f"[Code from {url}]\n```python\n{codes[0][:800]}\n```")
                        print(f"         Extracted code from {url[:50]}...")
                        break
    except Exception as e:
        print(f"         Web search failed: {e}")
    
    print("   [4/4] arXiv Search...")
    try:
        loader = ArxivLoader(
            query=query[:200],
            load_max_docs=2,
            load_all_available_meta=True
        )
        arxiv_docs = loader.load()
        if arxiv_docs:
            arxiv_text = "\n".join([
                f"[{doc.metadata.get('Title', 'Unknown')[:80]}]\n{doc.page_content[:500]}"
                for doc in arxiv_docs[:2]
            ])
            context_parts.append(f"[arXiv Papers]\n{arxiv_text}")
            print(f"         Found {len(arxiv_docs)} arXiv papers")
    except Exception as e:
        print(f"         arXiv search failed: {e}")
    
    return "\n\n".join(context_parts) if context_parts else "No external knowledge found."


def _parse_response(raw: str) -> dict:
    """Parse JSON response from LLM"""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(cleaned)
    except:
        pass
    
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    return {"hypothesis": {"title": "Continue optimization"}, "python_code": ""}


def _extract_code(result: dict) -> str:
    """Extract Python code from response"""
    code = result.get("python_code", "")
    
    if "```python" in code:
        match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    if "```" in code:
        match = re.search(r'```\s*(.*?)\s*```', code, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return code.strip()


def _review_assumptions(state: AgentState, knowledge_context: str, llm) -> dict:
    """Review assumptions and select one to relax"""
    checklist = state.get("assumptions_checklist", [])
    sim_output = state.get("simulation_output")
    
    current_conductivity = sim_output.conductivity if sim_output and sim_output.conductivity else 0.0
    target = 1.97e-6
    error_rate = abs(target - current_conductivity) / target * 100 if current_conductivity else 100.0
    error_message = sim_output.error_message if sim_output else "No previous result"
    
    prompt_vars = {
        "assumptions_checklist": format_assumptions_for_prompt(checklist),
        "current_conductivity": f"{current_conductivity:.2e}" if current_conductivity else "N/A",
        "error_rate": f"{error_rate:.1f}",
        "error_message": error_message or "No error",
        "knowledge_context": knowledge_context[:2000]
    }
    
    full_response = ""
    for chunk in llm.stream(ASSUMPTION_REVIEW_PROMPT.format_messages(**prompt_vars)):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        full_response += content
    
    return _parse_response(full_response)


def scientist_node(state: AgentState, knowledge_retriever, llm) -> dict:
    """Scientist: Entry point - analyzes results, decides END, generates improved code"""
    iteration = state.get("iteration_count", 0) + 1
    research_log = state.get("research_log", [])
    run_dir = state.get("run_dir")
    current_code = state.get("python_code", "")
    
    print(f"[Scientist] Iteration {iteration}")
    
    sim_output = state.get("simulation_output")
    
    if sim_output and sim_output.is_success and sim_output.conductivity:
        target = 1.97e-6
        error_rate = abs(target - sim_output.conductivity) / target * 100
        
        print(f"   Previous result: σ = {sim_output.conductivity} S/cm (error: {error_rate:.1f}%)")
        
        if error_rate < 10:
            print(f"   🎯 Target achieved! Error rate {error_rate:.1f}% < 10%")
            return {
                "status": "FINISH",
                "hypothesis": f"Target achieved with {error_rate:.1f}% error",
                "python_code": current_code,
                "iteration_count": iteration,
                "research_log": research_log + [f"Scientist: Target achieved ({error_rate:.1f}% error)"]
            }
    
    if iteration > Config.MAX_LOOPS:
        print(f"   Max iterations ({Config.MAX_LOOPS}) reached. Finishing.")
        return {
            "status": "FINISH",
            "hypothesis": "Max iterations reached",
            "python_code": current_code,
            "iteration_count": iteration,
            "research_log": research_log + ["Scientist: Max iterations reached"]
        }
    
    print("   Searching knowledge and generating improved code...")
    
    if sim_output:
        sim_result_text = f"""
Success: {sim_output.is_success}
Conductivity: {sim_output.conductivity} S/cm
Error: {sim_output.error_message or 'None'}
"""
    else:
        sim_result_text = "No previous simulation result (first iteration)."
    
    print("   [1/3] Reviewing assumptions...")
    assumptions_checklist = state.get("assumptions_checklist", [])
    discovered_gaps = state.get("discovered_gaps", [])
    active_assumptions = get_active_assumptions(assumptions_checklist)
    
    assumption_review = {}
    assumption_to_relax = None
    implementation_plan = ""
    assumption_context = "Continue improving the simulation."
    
    if active_assumptions:
        initial_context = format_assumptions_for_prompt(assumptions_checklist)
        assumption_review = _review_assumptions(state, initial_context, llm)
        selected_id = assumption_review.get("selected_assumption_id", "")
        implementation_plan = assumption_review.get("implementation_plan", "")
        
        if selected_id == "DISCOVERED" and assumption_review.get("discovered_gap"):
            new_gap = assumption_review["discovered_gap"]
            discovered_gaps.append(new_gap)
            print(f"   [!] Discovered new gap: {new_gap[:80]}...")
        else:
            for a in active_assumptions:
                if a["id"] == selected_id:
                    assumption_to_relax = a
                    print(f"   Selected: [{a['id']}] {a['name']}")
                    break
    
    if assumption_to_relax:
        query = f"{assumption_to_relax['name']} kMC simulation implementation {assumption_to_relax['description'][:100]}"
        
        assumptions_checklist = relax_assumption(
            assumptions_checklist,
            assumption_to_relax["id"],
            assumption_review.get("reason_to_relax", ""),
            assumption_review.get("physical_reality", ""),
            implementation_plan,
            iteration
        )
        
        assumption_context = f"""
[{assumption_to_relax['id']}] {assumption_to_relax['name']}
Reason: {assumption_review.get('reason_to_relax', '')}
Physical Reality: {assumption_review.get('physical_reality', '')}
"""
    else:
        query = f"{state['goal']} kMC ionic conductivity solid electrolyte simulation"
        implementation_plan = "General optimization"
    
    print(f"   [2/3] Searching knowledge (query: {query[:60]}...)")
    knowledge_context = _search_knowledge(query, run_dir, knowledge_retriever)
    
    print("   [3/3] Generating code...")
    
    prompt_vars = {
        "goal": state["goal"],
        "current_code": current_code[:3000] if current_code else "# No existing code",
        "simulation_result": sim_result_text,
        "knowledge_context": knowledge_context[:4000],
        "simulation_time": Config.SIMULATION_TIME,
        "assumption_being_relaxed": assumption_context,
        "implementation_plan": implementation_plan
    }
    
    full_response = ""
    for chunk in llm.stream(SCIENTIST_PROMPT.format_messages(**prompt_vars)):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
        full_response += content
    print("\n")
    
    result = _parse_response(full_response)
    
    hypothesis_data = result.get("hypothesis", {})
    if isinstance(hypothesis_data, dict):
        hypothesis_text = f"""## {hypothesis_data.get('title', 'Optimization')}
**Mechanism:** {hypothesis_data.get('mechanism', 'N/A')}
**Key Changes:** {', '.join(hypothesis_data.get('key_changes', []))}"""
    else:
        hypothesis_text = str(hypothesis_data)
    
    python_code = _extract_code(result)
    
    if not python_code:
        python_code = current_code
        print("   [Warning] No code generated, keeping previous code")
    else:
        print(f"   Generated {len(python_code)} bytes of code")
    
    focus_assumption_id = assumption_to_relax["id"] if assumption_to_relax else None
    
    return {
        "status": "CONTINUE",
        "hypothesis": hypothesis_text,
        "python_code": python_code,
        "iteration_count": iteration,
        "assumptions_checklist": assumptions_checklist,
        "current_focus_assumption": focus_assumption_id,
        "discovered_gaps": discovered_gaps,
        "research_log": research_log + [f"Scientist: {hypothesis_data.get('title', 'Generated code')}"]
    }


def create_scientist_node(knowledge_retriever, llm):
    def node_fn(state: AgentState) -> dict:
        return scientist_node(state, knowledge_retriever, llm)
    return node_fn
