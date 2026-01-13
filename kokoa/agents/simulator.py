"""
Simulator Agent - Code execution and result collection
"""

import os
import sys
import subprocess
import time
import textwrap
import re
import json
from datetime import datetime

from kokoa.config import Config
from kokoa.state import AgentState, SimulationResult


def validate_kmc_code(code: str) -> tuple:
    """Validate generated code is actually kMC simulation
    
    Returns: (is_valid: bool, error_message: str)
    """
    # Forbidden patterns (hallucination detection)
    forbidden = ["students", "friendships", "infected", "infection", "BFS", "breadth"]
    for pattern in forbidden:
        if pattern.lower() in code.lower():
            return False, f"Invalid code: contains '{pattern}' (not kMC simulation)"
    
    # Required patterns
    required = ["conductivity"]
    missing = [p for p in required if p.lower() not in code.lower()]
    if missing:
        return False, f"Missing required elements: {missing}"
    
    # Note: Non-ASCII characters like 'Å' (Angstrom), '·' are allowed in scientific code
    
    return True, "OK"


def execute_simulation_code(code: str, run_dir: str, iteration: int, timeout: int = 120) -> SimulationResult:
    sim_dir = os.path.join(run_dir, "simulation")
    os.makedirs(sim_dir, exist_ok=True)
    
    script_name = f"{iteration:03d}.py"
    script_path = os.path.abspath(os.path.join(sim_dir, script_name))
    
    indented_code = textwrap.indent(code, '    ')
    safe_run_dir = os.path.abspath(run_dir).replace("\\", "/")
    
    # Calculate project root and CIF path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cif_path = os.path.join(project_root, "Li4.47La3Zr2O12.cif").replace("\\", "/")
    safe_project_root = project_root.replace("\\", "/")
    
    wrapped_code = f'''"""
KOKOA Simulation #{iteration}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "{safe_project_root}"
_CIF_PATH = "{cif_path}"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('{safe_run_dir}')
except Exception as e:
    sys.stderr.write(f"Directory Error: {{e}}\\n")

try:
{indented_code}
except Exception as e:
    sys.stderr.write(f"Runtime Error: {{str(e)}}\\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(wrapped_code)
    
    print(f"   Code saved: simulation/{script_name}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=run_dir
        )
        
        stdout, stderr = result.stdout, result.stderr
        is_success = result.returncode == 0
        
        conductivity = None
        match = re.search(r"[Cc]onductivity[:\s]+([0-9.eE+-]+)", stdout)
        if match:
            try:
                conductivity = float(match.group(1))
            except ValueError:
                pass
        
        image_path = None
        if os.path.exists(os.path.join(run_dir, "result.png")):
            image_path = os.path.join(run_dir, "result.png")
        
        return SimulationResult(
            is_success=is_success,
            conductivity=conductivity,
            error_message=stderr if stderr else None,
            execution_log=f"[STDOUT]\n{stdout}\n[STDERR]\n{stderr}",
            image_path=image_path
        )
        
    except subprocess.TimeoutExpired:
        return SimulationResult(
            is_success=False,
            error_message=f"Timeout after {timeout}s",
            execution_log="Execution timed out"
        )
    except Exception as e:
        return SimulationResult(
            is_success=False,
            error_message=str(e),
            execution_log=f"Execution error: {e}"
        )


def save_result(result: SimulationResult, run_dir: str, iteration: int):
    result_dir = os.path.join(run_dir, "simulation_result")
    os.makedirs(result_dir, exist_ok=True)
    
    result_name = f"{iteration:03d}.json"
    result_path = os.path.join(result_dir, result_name)
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "is_success": result.is_success,
        "conductivity": result.conductivity,
        "conductivity_unit": "S/cm",
        "error_message": result.error_message,
        "execution_log": result.execution_log,
        "image_path": result.image_path
    }
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"   Result saved: simulation_result/{result_name}")


def simulator_node(state: AgentState) -> dict:    
    iteration = state.get("iteration_count", 0) + 1
    code = state.get("python_code", "")
    run_dir = state.get("run_dir")
    research_log = state.get("research_log", [])
    
    if not code:
        return {
            "simulation_output": SimulationResult(
                is_success=False,
                error_message="No code to execute",
                execution_log="Empty code"
            ),
            "iteration_count": iteration,
            "research_log": research_log + ["Simulator: No code"]
        }
    
    # Validate code before execution
    is_valid, validation_msg = validate_kmc_code(code)
    if not is_valid:
        print(f"   ⚠️ Code validation failed: {validation_msg}")
        # Rollback to last valid code
        last_valid = state.get("last_valid_code", "")
        if last_valid:
            print("   → Rolling back to last valid code")
            code = last_valid
        else:
            return {
                "simulation_output": SimulationResult(
                    is_success=False,
                    error_message=f"Code validation failed: {validation_msg}",
                    execution_log="Invalid code detected"
                ),
                "iteration_count": iteration,
                "research_log": research_log + [f"Simulator: Validation failed - {validation_msg}"]
            }
    
    if not run_dir:
        run_dir = os.path.join(Config.RUNS_DIR, "default")
        os.makedirs(run_dir, exist_ok=True)
    
    result = execute_simulation_code(code, run_dir, iteration)
    save_result(result, run_dir, iteration)
    
    if result.is_success:
        log_msg = f"Simulator: σ = {result.conductivity} S/cm"
    else:
        log_msg = f"Simulator: Failed - {result.error_message}"
    
    print(f"   → {log_msg}")
    
    return {
        "simulation_output": result,
        "iteration_count": iteration,
        "research_log": research_log + [log_msg]
    }
