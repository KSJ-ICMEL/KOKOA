"""
KOKOA State Definitions
"""

import os
import json
import shutil
from typing import Optional, List, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field


class FocusedAssumptionSummary(TypedDict):
    id: str
    name: str
    status: str
    reason_to_relax: Optional[str]
    physical_reality: Optional[str]
    implementation_plan: Optional[str]

from kokoa.config import Config
from kokoa.assumptions_config import AssumptionItem, BASE_ASSUMPTIONS


class SimulationResult(BaseModel):
    is_success: bool = Field(...)
    conductivity: Optional[float] = Field(None)
    error_message: Optional[str] = Field(None)
    execution_log: str = Field(...)
    image_path: Optional[str] = Field(None)
    log_error: Optional[float] = Field(None)


class AgentState(TypedDict):
    # Core
    goal: str
    hypothesis: str
    current_code: str
    previous_code: str  # For diff generation in Archivist

    # Simulation
    simulation_output: Optional[SimulationResult]
    current_log_error: Optional[float]
    conductivity_history: List[dict]
    
    # Flow control
    research_log: List[str]
    status: str
    iteration_count: int
    
    # Assumptions
    assumptions_checklist: List[AssumptionItem]
    relaxed_hurdles: List[dict] # [{id, name, reason, changes}, ...]
    target_assumption: Optional[str] # Single assumption targeted in this run
    
    # Runtime
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
        image_path=data.get("image_path"),
        log_error=data.get("log_error")
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
    
    import copy
    assumptions = copy.deepcopy(BASE_ASSUMPTIONS)
    
    return {
        # Core
        "goal": goal,
        "hypothesis": "",
        "previous_code": initial_code,
        "current_code": "",

        # Simulation
        "simulation_output": initial_result,
        "current_log_error": initial_result.log_error if initial_result and initial_result.log_error is not None else 10.0,
        "conductivity_history": [],
        
        # Flow control
        "research_log": [f"--- Run {run_id} Started ---"],
        "status": "running",
        "iteration_count": 0,
        
        # Assumptions
        "assumptions_checklist": assumptions,
        "relaxed_hurdles": [],
        "target_assumption": "",
        
        # Runtime
        "run_id": run_id,
        "run_dir": run_dir,
    }
