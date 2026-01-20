"""
KOKOA - Knowledge-Oriented Kinetic Optimization Agent
======================================================
고체전해질 시뮬레이션 최적화를 위한 멀티 에이전트 시스템

3-Agent Architecture:
- Scientist: 지식 검색, 가설 생성, 코드 작성, 종료 판단
- CodeAgent: 시뮬레이션 실행 + 디버깅
- Archivist: 지식 아카이빙
"""

from kokoa.config import Config
from kokoa.state import AgentState, SimulationResult, create_initial_state
from kokoa.graph import build_workflow, run_experiment, visualize, save_graph_png
from kokoa.agents import (
    scientist_node, create_scientist_node,
    code_agent_node, create_code_agent_node,
    archivist_node, create_archivist_node,
)

__version__ = "0.2.0"
__all__ = [
    # Config
    "Config",
    # State
    "AgentState",
    "SimulationResult",
    "create_initial_state",
    # Graph
    "build_workflow",
    "run_experiment",
    "visualize",
    "save_graph_png",
    # Agents
    "scientist_node",
    "create_scientist_node",
    "code_agent_node",
    "create_code_agent_node",
    "archivist_node",
    "create_archivist_node",
]
