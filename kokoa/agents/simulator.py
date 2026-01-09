"""
Simulator Agent
===============
코드를 샌드박스에서 실행하고 결과 수집
코드 변천사 추적을 위해 넘버링된 파일로 저장
"""

import os
import sys
import subprocess
import time
import textwrap
import re
from glob import glob

from kokoa.config import Config
from kokoa.state import AgentState, SimulationResult


def get_next_script_number() -> int:
    """다음 스크립트 번호 반환 (기존 파일들 확인)"""
    pattern = os.path.join(Config.WORKSPACE_DIR, "simulation_*.py")
    existing = glob(pattern)
    
    if not existing:
        return 1
    
    numbers = []
    for path in existing:
        basename = os.path.basename(path)
        match = re.search(r"simulation_(\d+)\.py", basename)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) + 1 if numbers else 1


def execute_simulation_code(code: str, iteration: int = None, timeout: int = 60) -> SimulationResult:
    """
    시뮬레이션 코드 실행
    
    Args:
        code: 실행할 Python 코드
        iteration: 반복 번호 (None이면 자동 계산)
        timeout: 타임아웃 (초)
    
    Returns:
        SimulationResult
    
    Note:
        - initial_state.py는 절대 덮어쓰지 않음
        - 코드는 simulation_XXX.py 형식으로 저장
    """
    os.makedirs(Config.WORKSPACE_DIR, exist_ok=True)
    
    if iteration is None:
        iteration = get_next_script_number()
    
    script_name = f"simulation_{iteration:03d}.py"
    script_path = os.path.join(Config.WORKSPACE_DIR, script_name)
    
    indented_code = textwrap.indent(code, '    ')
    safe_workspace = os.path.abspath(Config.WORKSPACE_DIR).replace("\\", "/")
    
    wrapped_code = f'''"""
KOKOA Generated Simulation Script #{iteration}
Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

import os
import sys
import traceback

try:
    os.chdir('{safe_workspace}')
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
    
    print(f"   💾 코드 저장: {script_name}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Config.WORKSPACE_DIR
        )
        
        stdout = result.stdout
        stderr = result.stderr
        is_success = result.returncode == 0
        
        conductivity = None
        cond_match = re.search(r"[Cc]onductivity[:\s]+([0-9.eE+-]+)", stdout)
        if cond_match:
            try:
                conductivity = float(cond_match.group(1))
            except ValueError:
                pass
        
        image_path = None
        if os.path.exists(os.path.join(Config.WORKSPACE_DIR, "result.png")):
            image_path = os.path.join(Config.WORKSPACE_DIR, "result.png")
        
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
            error_message=f"Timeout after {timeout} seconds",
            execution_log="Execution timed out"
        )
    except Exception as e:
        return SimulationResult(
            is_success=False,
            error_message=str(e),
            execution_log=f"Execution error: {e}"
        )


def run_initial_state(timeout: int = 120) -> SimulationResult:
    """initial_state.py 실행 (첫 번째 시뮬레이션)"""
    script_path = os.path.join(Config.WORKSPACE_DIR, "initial_state.py")
    
    if not os.path.exists(script_path):
        return SimulationResult(
            is_success=False,
            error_message=f"initial_state.py not found in {Config.WORKSPACE_DIR}",
            execution_log="Missing initial simulation script"
        )
    
    print(f"   📄 실행: initial_state.py")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Config.WORKSPACE_DIR
        )
        
        stdout = result.stdout
        stderr = result.stderr
        is_success = result.returncode == 0
        
        conductivity = None
        cond_match = re.search(r"[Cc]onductivity[:\s]+([0-9.eE+-]+)", stdout)
        if cond_match:
            try:
                conductivity = float(cond_match.group(1))
            except ValueError:
                pass
        
        return SimulationResult(
            is_success=is_success,
            conductivity=conductivity,
            error_message=stderr if stderr else None,
            execution_log=f"[STDOUT]\n{stdout}\n[STDERR]\n{stderr}",
            image_path=None
        )
        
    except subprocess.TimeoutExpired:
        return SimulationResult(
            is_success=False,
            error_message=f"Timeout after {timeout} seconds",
            execution_log="Execution timed out"
        )
    except Exception as e:
        return SimulationResult(
            is_success=False,
            error_message=str(e),
            execution_log=f"Execution error: {e}"
        )


def save_result(result: SimulationResult, iteration: int, script_name: str = None):
    """시뮬레이션 결과를 JSON 파일로 저장"""
    import json
    from datetime import datetime
    
    if script_name:
        result_name = script_name.replace(".py", "_result.json")
    else:
        result_name = f"simulation_{iteration:03d}_result.json"
    
    result_path = os.path.join(Config.WORKSPACE_DIR, result_name)
    
    result_data = {
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
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 결과 저장: {result_name}")
    return result_path


def simulator_node(state: AgentState) -> dict:
    print("⚗️ [Simulator] 시뮬레이션 실행 중...")
    
    iteration = state.get("iteration_count", 0)
    code = state.get("python_code", "")
    research_log = state.get("research_log", [])
    
    if iteration == 0 and not code:
        print("   → 초기 시뮬레이션 (initial_state.py) 실행")
        result = run_initial_state()
        script_name = "initial_state.py"
        
        if result.is_success:
            log_msg = f"Simulator: Initial run. Conductivity = {result.conductivity} S/cm"
        else:
            log_msg = f"Simulator: Initial run failed. Error = {result.error_message}"
        
        save_result(result, iteration, script_name)
    
    elif not code:
        return {
            "simulation_output": SimulationResult(
                is_success=False,
                error_message="No code to execute",
                execution_log="Empty code"
            ),
            "iteration_count": iteration + 1,
            "research_log": research_log + ["Simulator: No code to execute"]
        }
    
    else:
        result = execute_simulation_code(code, iteration=iteration)
        script_name = f"simulation_{iteration:03d}.py"
        
        if result.is_success:
            log_msg = f"Simulator: Success. Conductivity = {result.conductivity} S/cm"
        else:
            log_msg = f"Simulator: Failed. Error = {result.error_message}"
        
        save_result(result, iteration, script_name)
    
    print(f"   → {log_msg[:60]}...")
    
    return {
        "simulation_output": result,
        "iteration_count": iteration + 1,
        "research_log": research_log + [log_msg]
    }
