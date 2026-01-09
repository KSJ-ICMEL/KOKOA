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
    ("system", """You are an expert scientist in computational materials science and solid-state batteries.
Your goal is to formulate a novel and scientifically grounded hypothesis.

**CRITICAL**: First, evaluate if the provided context is SUFFICIENT to formulate a concrete hypothesis.

If INSUFFICIENT:
- Set "needs_research" to true
- Provide a specific "research_query" for arXiv search
- Describe the "knowledge_gap"
- Leave "hypothesis" empty

If SUFFICIENT:
- Set "needs_research" to false
- Formulate a detailed hypothesis

Strategy:
1. Retrieve & Analyze: Use academic context to find known mechanisms.
2. Core Property: Identify mechanism that enhances primary target.
3. Synergy: Identify second mechanism addressing trade-offs.

Output JSON only:
{{
    "needs_research": false,
    "research_query": "",
    "knowledge_gap": "",
    "hypothesis": "## Title\\n**Core Mechanism:** ...\\n**Synergistic Strategy:** ...\\n**Expected Outcome:** ...\\n**Justification:** ..."
}}
"""),
    ("user", """
Goal: {goal}
Feedback: {feedback}
Failed Attempts: {failed_attempts}
Research Attempts: {research_attempts}/{max_research_attempts}

[Academic Context]:
{rag_context}

Output JSON only.
""")
])


def theorist_node(state: AgentState, knowledge_retriever, llm) -> dict:
    print("💡 [Theorist] RAG 검색 및 가설 수립 중...")
    
    research_attempts = state.get("research_attempts", 0)
    max_attempts = Config.MAX_RESEARCH_ATTEMPTS
    research_log = state.get("research_log", [])
    
    if research_attempts >= max_attempts:
        print(f"   ⚠️ 최대 연구 시도 횟수 도달 ({max_attempts})")
    
    query = f"{state['goal']} optimization mechanisms solid electrolyte"
    retrieved_docs = knowledge_retriever.invoke(query)
    rag_text = "\n\n".join([
        f"[Paper {i+1}] {doc.page_content[:500]}..." 
        for i, doc in enumerate(retrieved_docs)
    ]) or "No relevant documents found."
    
    chain = THEORIST_PROMPT | llm | JsonOutputParser()
    
    try:
        result = chain.invoke({
            "goal": state["goal"],
            "feedback": state.get("user_feedback") or "None",
            "failed_attempts": ", ".join(state.get("failed_attempts", [])) or "None",
            "research_attempts": research_attempts,
            "max_research_attempts": max_attempts,
            "rag_context": rag_text
        })
    except Exception as e:
        print(f"   ⚠️ JSON 파싱 실패: {e}")
        result = {"needs_research": False, "hypothesis": f"Optimize {state['goal']} through parameter variation."}
    
    needs_research = result.get("needs_research", False) and research_attempts < max_attempts
    
    if needs_research:
        print(f"   → 외부 연구 필요: {result.get('knowledge_gap', '')[:50]}")
        return {
            "needs_research": True,
            "research_query": result.get("research_query", state["goal"]),
            "knowledge_gap": result.get("knowledge_gap", ""),
            "research_log": research_log + [f"Theorist: Requesting research"]
        }
    
    hypothesis = result.get("hypothesis", f"Optimize {state['goal']}")
    first_line = hypothesis.split('\n')[0][:60]
    print(f"   → 가설: {first_line}...")
    
    return {
        "hypothesis": hypothesis,
        "needs_research": False,
        "research_log": research_log + [f"Theorist: {first_line}"]
    }


def create_theorist_node(knowledge_retriever, llm):
    def node_fn(state: AgentState) -> dict:
        return theorist_node(state, knowledge_retriever, llm)
    return node_fn
