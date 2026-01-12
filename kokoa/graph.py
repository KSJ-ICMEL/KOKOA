"""
KOKOA Graph Assembly
====================
에이전트들을 연결하는 LangGraph 워크플로우
"""

import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from kokoa.config import Config
from kokoa.state import AgentState, create_initial_state
from kokoa.agents.researcher import researcher_node


def analyst_router(state: AgentState) -> str:
    status = state.get("status", "")
    if status == "FINISH":
        print("🎉 [System] 연구 목표 달성! 종료합니다.")
        return "end"
    elif status in ("RETRY", "ROLLBACK"):
        print(f"🔄 [System] {status}. Theorist에게 피드백 전달...")
        return "theorist"
    return "theorist"


def theorist_router(state: AgentState) -> str:
    needs_research = state.get("needs_research", False)
    research_attempts = state.get("research_attempts", 0)
    
    if needs_research and research_attempts < Config.MAX_RESEARCH_ATTEMPTS:
        print("📚 [System] 외부 연구 필요. Researcher 호출...")
        return "researcher"
    
    if needs_research:
        print(f"⚠️ [System] 최대 연구 시도 도달. Engineer로 진행...")
    
    return "engineer"


def build_workflow(theorist_node, engineer_node, simulator_node, analyst_node):
    """
    KOKOA 워크플로우 빌드
    
    Args:
        theorist_node: create_theorist_node()로 생성된 노드
        engineer_node: create_engineer_node()로 생성된 노드
        simulator_node: simulator_node 함수
        analyst_node: create_analyst_node()로 생성된 노드
    
    Returns:
        compiled LangGraph app
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Theorist", theorist_node)
    workflow.add_node("Engineer", engineer_node)
    workflow.add_node("Simulator", simulator_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Researcher", researcher_node)
    
    workflow.add_conditional_edges(
        "Theorist",
        theorist_router,
        {"researcher": "Researcher", "engineer": "Engineer"}
    )
    
    workflow.add_edge("Researcher", "Theorist")
    workflow.add_edge("Engineer", "Simulator")
    workflow.add_edge("Simulator", "Analyst")
    
    workflow.add_conditional_edges(
        "Analyst",
        analyst_router,
        {"theorist": "Theorist", "end": END}
    )
    
    workflow.set_entry_point("Analyst")
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


def run_experiment(app, goal: str, thread_id: str = None):
    """
    실험 실행
    
    Args:
        app: build_workflow()로 빌드된 앱
        goal: 연구 목표
        thread_id: 스레드 ID (None이면 자동 생성)
    
    Returns:
        최종 상태
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())[:8]
    
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(goal)
    
    run_dir = initial_state.get("run_dir", "unknown")
    run_id = initial_state.get("run_id", thread_id)
    
    print(f"🚀 KOKOA 시작 (Run: {run_id})")
    print(f"📁 출력 디렉토리: {run_dir}")
    print(f"🎯 목표: {goal[:80]}...")
    print("=" * 60)
    
    final_state = None
    try:
        for event in app.stream(initial_state, config):
            for node_name, node_output in event.items():
                print(f"\n📍 [{node_name}] 완료")
                
                if node_name == "Theorist":
                    hyp = node_output.get('hypothesis', '')[:100]
                    if hyp:
                        print(f"   💡 가설: {hyp}...")
                elif node_name == "Engineer":
                    code_len = len(node_output.get('python_code', ''))
                    print(f"   💾 코드: {code_len} bytes")
                elif node_name == "Simulator":
                    result = node_output.get("simulation_output")
                    if result:
                        print(f"   🧪 결과: Success={result.is_success}, Cond={result.conductivity}")
                elif node_name == "Analyst":
                    status = node_output.get("status")
                    err = node_output.get('current_error_rate', 0)
                    print(f"   📊 판단: {status} (오차율: {err:.2f}%)")
                elif node_name == "Researcher":
                    attempts = node_output.get("research_attempts", 0)
                    print(f"   📚 연구 시도: {attempts}")
                
                final_state = node_output
        
        print("\n" + "=" * 60)
        print("🏁 실험 종료")
        
    except Exception as e:
        print(f"\n🚨 에러: {e}")
    
    return final_state


def visualize(app):
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:
        print("시각화 불가 (IPython 환경 필요)")
