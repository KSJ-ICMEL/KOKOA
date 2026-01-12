"""
Theorist Agent
==============
RAG 기반 가설 수립 + 외부 연구 요청 기능
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from kokoa.config import Config
from kokoa.state import AgentState


THEORIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert scientist in computational materials science and solid-state battery simulation.

**IMPORTANT GUIDELINES**:
1. **INCREMENTAL CHANGES ONLY**: Do NOT try to modify too many things at once. Focus on ONE small, testable improvement per iteration.
2. **USE EXISTING KNOWLEDGE FIRST**: The academic context provided contains valuable information. Extract and use it thoroughly before requesting external research.
3. **RESEARCH AS LAST RESORT**: Only set needs_research=true if the provided context is genuinely insufficient. The knowledge base has been carefully curated.

**YOUR TASK**:
Based on the current simulation code and results, formulate hypotheses about what NEW FACTORS to add or EXISTING FACTORS to modify to make the simulation more realistic.

**PROCESS**:
1. Analyze the provided academic context deeply. Look for:
   - Physical mechanisms not yet in the simulation
   - Parameter values from literature
   - Correction factors or correlations
   
2. Generate exactly 3 hypotheses, then RANK them by feasibility and expected impact.

3. Pass ONLY the top-ranked hypothesis to the Engineer.

**OUTPUT FORMAT (JSON only)**:
{{
    "needs_research": false,
    "research_query": "",
    "knowledge_gap": "",
    "hypotheses": [
        {{"rank": 1, "title": "...", "mechanism": "...", "expected_improvement": "...", "implementation_complexity": "low/medium/high"}},
        {{"rank": 2, "title": "...", "mechanism": "...", "expected_improvement": "...", "implementation_complexity": "..."}},
        {{"rank": 3, "title": "...", "mechanism": "...", "expected_improvement": "...", "implementation_complexity": "..."}}
    ],
    "selected_hypothesis": "## [Title of Rank 1]\\n**Mechanism:** ...\\n**Implementation:** Specific code changes...\\n**Expected Outcome:** ..."
}}
"""),
    ("user", """
Goal: {goal}
Previous Feedback: {feedback}
Failed Attempts: {failed_attempts}
Research Attempts: {research_attempts}/{max_research_attempts}

[Current Simulation Code]:
```python
{current_code}
```

[Academic Context - USE THIS THOROUGHLY]:
{rag_context}

Generate 3 ranked hypotheses. Output JSON only.
""")
])


def theorist_node(state: AgentState, knowledge_retriever, llm) -> dict:
    print("[Theorist] Analyzing and formulating hypotheses...")
    
    research_attempts = state.get("research_attempts", 0)
    max_attempts = Config.MAX_RESEARCH_ATTEMPTS
    research_log = state.get("research_log", [])
    current_code = state.get("python_code", "")
    
    if research_attempts >= max_attempts:
        print(f"   Max research attempts reached ({max_attempts})")
    
    query = f"{state['goal']} kMC simulation ionic conductivity solid electrolyte"
    retrieved_docs = knowledge_retriever.invoke(query)
    rag_text = "\n\n".join([
        f"[Paper {i+1}] {doc.page_content[:800]}" 
        for i, doc in enumerate(retrieved_docs)
    ]) or "No relevant documents found."
    
    code_preview = current_code[:2000] if len(current_code) > 2000 else current_code
    
    chain = THEORIST_PROMPT | llm | JsonOutputParser()
    
    try:
        result = chain.invoke({
            "goal": state["goal"],
            "feedback": state.get("user_feedback") or "None",
            "failed_attempts": ", ".join(state.get("failed_attempts", [])) or "None",
            "research_attempts": research_attempts,
            "max_research_attempts": max_attempts,
            "current_code": code_preview or "(No code yet)",
            "rag_context": rag_text
        })
    except Exception as e:
        print(f"   JSON parsing failed: {e}")
        result = {"needs_research": False, "selected_hypothesis": f"Optimize {state['goal']} through parameter variation."}
    
    needs_research = result.get("needs_research", False) and research_attempts < max_attempts
    
    if needs_research:
        gap = result.get('knowledge_gap', '')[:50]
        print(f"   -> Research needed: {gap}")
        return {
            "needs_research": True,
            "research_query": result.get("research_query", state["goal"]),
            "knowledge_gap": result.get("knowledge_gap", ""),
            "research_log": research_log + [f"Theorist: Requesting research - {gap}"]
        }
    
    hypotheses = result.get("hypotheses", [])
    if hypotheses:
        print(f"   -> Generated {len(hypotheses)} hypotheses:")
        for h in hypotheses[:3]:
            print(f"      #{h.get('rank', '?')}: {h.get('title', 'Untitled')[:40]}...")
    
    hypothesis = result.get("selected_hypothesis", result.get("hypothesis", f"Optimize {state['goal']}"))
    first_line = hypothesis.split('\n')[0][:60]
    print(f"   -> Selected: {first_line}...")
    
    return {
        "hypothesis": hypothesis,
        "needs_research": False,
        "research_log": research_log + [f"Theorist: {first_line}"]
    }


def create_theorist_node(knowledge_retriever, llm):
    def node_fn(state: AgentState) -> dict:
        return theorist_node(state, knowledge_retriever, llm)
    return node_fn
