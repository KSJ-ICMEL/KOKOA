"""
KOKOA State Definitions
"""

import os
from typing import Optional, List, TypedDict
from pydantic import BaseModel, Field


class SimulationResult(BaseModel):
    is_success: bool = Field(..., description="시뮬레이션 성공 여부")
    conductivity: Optional[float] = Field(None, description="계산된 전도도 (S/cm)")
    error_message: Optional[str] = Field(None, description="에러 메시지")
    execution_log: str = Field(..., description="실행 로그")
    image_path: Optional[str] = Field(None, description="생성된 이미지 경로")


class AgentState(TypedDict):
    goal: str
    hypothesis: str
    python_code: str
    
    simulation_output: Optional[SimulationResult]
    current_error_rate: Optional[float]
    
    research_log: List[str]
    status: str
    user_feedback: Optional[str]
    failed_attempts: List[str]
    iteration_count: int
    
    needs_research: bool
    research_query: Optional[str]
    knowledge_gap: Optional[str]
    research_attempts: int


def load_initial_code(workspace_dir: str = "./workspace") -> str:
    """initial_state.py 코드 로드 (Engineer가 수정 기반으로 사용)"""
    initial_path = os.path.join(workspace_dir, "initial_state.py")
    
    if os.path.exists(initial_path):
        with open(initial_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def create_initial_state(goal: str, load_code: bool = False, workspace_dir: str = "./workspace") -> AgentState:
    """
    초기 상태 생성
    
    Args:
        goal: 연구 목표
        load_code: True면 initial_state.py 코드를 python_code에 로드
        workspace_dir: workspace 디렉토리 경로
    """
    initial_code = ""
    if load_code:
        initial_code = load_initial_code(workspace_dir)
    
    return {
        "goal": goal,
        "hypothesis": "",
        "python_code": initial_code,
        "simulation_output": None,
        "current_error_rate": 100.0,
        "research_log": ["--- Research Log Started ---"],
        "failed_attempts": [],
        "iteration_count": 0,
        "status": "running",
        "user_feedback": None,
        "needs_research": False,
        "research_query": None,
        "knowledge_gap": None,
        "research_attempts": 0,
    }

