"""
Scientist Agent - Computational Materials Scientist + Researcher
================================================================
Theorist와 Researcher를 통합한 에이전트
- 내부 지식 (RAG + Memory) 검색
- 외부 지식 (Tavily + arXiv) 검색
- 버전업된 시뮬레이션 코드 생성
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


SCIENTIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a **Computational Materials Scientist** specializing in kinetic Monte Carlo (kMC) simulations for solid-state electrolytes.

**YOUR ROLE:**
Analyze the current simulation code and results, then generate an improved version based on:
1. Your deep knowledge of solid-state physics and ionic conductivity
2. Academic papers (provided in context)
3. Web search results and code examples (provided in context)
4. Past experiments from memory (provided in context)

**CODE REQUIREMENTS:**
1. NO 'kokoa' imports - code must be standalone
2. Use: numpy, scipy, pymatgen, matplotlib
3. Use `_CIF_PATH` variable for CIF file (injected at runtime)
4. target_time = {simulation_time} seconds
5. Supercell: structure.make_supercell([3, 3, 3])
6. Print: `print(f"Conductivity: {{val}} S/cm")`
7. Print progress every 2000 steps

**IMPORTANT:**
- Focus on PHYSICS correctness
- Make ONE meaningful improvement per iteration
- The CodeAgent will fix any Python bugs later

**OUTPUT FORMAT (JSON only):**
{{
    "hypothesis": {{
        "title": "One-line improvement title",
        "mechanism": "Scientific explanation of what you're changing and why",
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

Generate improved kMC simulation code. Output JSON only.""")
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
    
    query = f"{state['goal']} kMC ionic conductivity solid electrolyte simulation"
    
    knowledge_context = _search_knowledge(query, run_dir, knowledge_retriever)
    
    prompt_vars = {
        "goal": state["goal"],
        "current_code": current_code[:3000] if current_code else "# No existing code",
        "simulation_result": sim_result_text,
        "knowledge_context": knowledge_context[:4000],
        "simulation_time": Config.SIMULATION_TIME
    }
    
    print("   Generating improved code...")
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
    
    return {
        "status": "CONTINUE",
        "hypothesis": hypothesis_text,
        "python_code": python_code,
        "iteration_count": iteration,
        "research_log": research_log + [f"Scientist: {hypothesis_data.get('title', 'Generated code')}"]
    }


def create_scientist_node(knowledge_retriever, llm):
    def node_fn(state: AgentState) -> dict:
        return scientist_node(state, knowledge_retriever, llm)
    return node_fn
