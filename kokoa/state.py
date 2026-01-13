"""
KOKOA State Definitions
"""

import os
import json
import shutil
from typing import Optional, List, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field

from kokoa.config import Config


class SimulationResult(BaseModel):
    is_success: bool = Field(...)
    conductivity: Optional[float] = Field(None)
    error_message: Optional[str] = Field(None)
    execution_log: str = Field(...)
    image_path: Optional[str] = Field(None)


class AgentState(TypedDict):
    goal: str
    hypothesis: str
    python_code: str
    last_valid_code: str  # For rollback support
    
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
    
    run_id: str
    run_dir: str


def load_initial_result() -> Optional[SimulationResult]:
    """Load pre-run initial_state result from initial_state/initial_state.json"""
    result_path = os.path.join(Config.INITIAL_STATE_DIR, "initial_state.json")
    
    if not os.path.exists(result_path):
        return None
    
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return SimulationResult(
        is_success=data.get("is_success", False),
        conductivity=data.get("conductivity"),
        error_message=data.get("error_message"),
        execution_log=data.get("execution_log", ""),
        image_path=data.get("image_path")
    )


def load_initial_code() -> str:
    """Load initial_state.py code"""
    code_path = os.path.join(Config.INITIAL_STATE_DIR, "initial_state.py")
    
    if os.path.exists(code_path):
        with open(code_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def create_run_directory(run_id: str) -> str:
    """Create run-specific directory with subdirectories"""
    run_dir = os.path.join(Config.RUNS_DIR, run_id)
    
    os.makedirs(os.path.join(run_dir, "simulation"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "simulation_result"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "chroma_store"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "pdf"), exist_ok=True)
    
    initial_chroma = Config.PERSIST_DIRECTORY
    if os.path.exists(initial_chroma):
        run_chroma = os.path.join(run_dir, "chroma_store")
        if not os.listdir(run_chroma):
            shutil.copytree(initial_chroma, run_chroma, dirs_exist_ok=True)
    
    return run_dir


def create_initial_state(goal: str, run_id: str = None) -> AgentState:
    """
    Create initial state with run-specific directory
    
    Args:
        goal: Research goal
        run_id: Run ID (auto-generated if None)
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = create_run_directory(run_id)
    
    initial_code = load_initial_code()
    initial_result = load_initial_result()
    
    return {
        "goal": goal,
        "hypothesis": "",
        "python_code": initial_code,
        "simulation_output": initial_result,
        "current_error_rate": 100.0,
        "research_log": [f"--- Run {run_id} Started ---"],
        "failed_attempts": [],
        "iteration_count": 0,
        "status": "running",
        "user_feedback": None,
        "needs_research": False,
        "research_query": None,
        "knowledge_gap": None,
        "research_attempts": 0,
        "run_id": run_id,
        "run_dir": run_dir,
    }
