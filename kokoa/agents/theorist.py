"""
Theorist Agent - RAG-based hypothesis generation
"""

import re
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from kokoa.config import Config
from kokoa.state import AgentState

USER_PROMPT = """
Goal: {goal}
Feedback: {feedback}
Failed: {failed_attempts}

Code:
```python
{current_code}
```

Context:
{rag_context}

Generate hypothesis. JSON only.
"""

THEORIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a computational materials scientist specializing in solid-state battery simulation.

TASK: Generate ONE testable hypothesis to improve kMC ionic conductivity simulation.

RULES:
1. ONE small change per iteration. No complete rewrites.
2. Use academic context provided. Research only if context insufficient.
3. Focus on physics: activation energies, hopping rates, Coulomb interactions.

OUTPUT (JSON only):
{{
    "needs_research": false,
    "research_query": "",
    "hypotheses": [
        {{"rank": 1, "title": "...", "mechanism": "...", "expected_improvement": "...", "implementation_complexity": "low/medium/high"}}
    ],
    "selected_hypothesis": "## Title\\n**Mechanism:** ...\\n**Implementation:** Specific code changes...\\n**Expected Outcome:** ..."
}}"""),
    ("user", USER_PROMPT)
])


def parse_json_response(raw: str) -> dict:
    """Robust JSON parsing with fallbacks"""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(cleaned)
    except:
        pass
    
    # Try to find JSON in response
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    return {"needs_research": False, "selected_hypothesis": "Continue parameter tuning."}


def theorist_node(state: AgentState, knowledge_retriever, llm) -> dict:
    research_attempts = state.get("research_attempts", 0)
    max_attempts = Config.MAX_RESEARCH_ATTEMPTS
    research_log = state.get("research_log", [])
    current_code = state.get("python_code", "")
    
    # RAG retrieval
    query = f"{state['goal']} kMC ionic conductivity"
    retrieved_docs = knowledge_retriever.invoke(query)
    rag_text = "\n".join([
        f"[{i+1}] {doc.page_content[:600]}" 
        for i, doc in enumerate(retrieved_docs[:2])
    ]) or "No documents."
    
    user_prompt_vars = {
        "goal": state["goal"],
        "feedback": state.get("user_feedback", "None"),
        "failed_attempts": ", ".join(state.get("failed_attempts", [])) or "None",
        "current_code": current_code,
        "rag_context": rag_text
    }
    
    # Print USER PROMPT
    print("[Theorist] USER PROMPT:")
    print(USER_PROMPT.format(**user_prompt_vars))
    
    # Stream LLM response with output
    print("[Theorist] RESPONSE:")
    full_response = ""
    for chunk in llm.stream(THEORIST_PROMPT.format_messages(**user_prompt_vars)):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
        full_response += content
    print("\n")
    
    result = parse_json_response(full_response)
    
    needs_research = result.get("needs_research", False) and research_attempts < max_attempts
    
    if needs_research:
        gap = result.get('knowledge_gap', '')
        print(f"   → Research needed: {gap}")
        return {
            "needs_research": True,
            "research_query": result.get("research_query", state["goal"]),
            "knowledge_gap": result.get("knowledge_gap", ""),
            "research_log": research_log + [f"Theorist: Research - {gap}"]
        }
    
    hypothesis = result.get("selected_hypothesis", "Continue optimization.")
    
    return {
        "hypothesis": hypothesis,
        "needs_research": False,
        "research_log": research_log + [f"Theorist: {hypothesis}"]
    }


def create_theorist_node(knowledge_retriever, llm):
    def node_fn(state: AgentState) -> dict:
        return theorist_node(state, knowledge_retriever, llm)
    return node_fn
