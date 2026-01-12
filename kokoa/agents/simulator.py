"""
Simulator Agent
===============
코드를 샌드박스에서 실행하고 결과 수집
simulation/ 폴더에 코드, simulation_result/ 폴더에 결과 저장
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


def execute_simulation_code(code: str, run_dir: str, iteration: int, timeout: int = 120) -> SimulationResult:
    sim_dir = os.path.join(run_dir, "simulation")
    os.makedirs(sim_dir, exist_ok=True)
    
    script_name = f"{iteration:03d}.py"
    script_path = os.path.join(sim_dir, script_name)
    
    indented_code = textwrap.indent(code, '    ')
    safe_run_dir = os.path.abspath(run_dir).replace("\\", "/")
    
    wrapped_code = f'''"""
KOKOA Simulation #{iteration}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
import os, sys, traceback

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
    print("[Simulator] Running simulation...")
    
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
            "research_log": research_log + ["Simulator: No code to execute"]
        }
    
    if not run_dir:
        run_dir = os.path.join(Config.RUNS_DIR, "default")
        os.makedirs(run_dir, exist_ok=True)
    
    result = execute_simulation_code(code, run_dir, iteration)
    save_result(result, run_dir, iteration)
    
    if result.is_success:
        log_msg = f"Simulator: Success. Conductivity = {result.conductivity} S/cm"
    else:
        log_msg = f"Simulator: Failed. Error = {result.error_message}"
    
    print(f"   -> {log_msg[:60]}...")
    
    return {
        "simulation_output": result,
        "iteration_count": iteration,
        "research_log": research_log + [log_msg]
    }
