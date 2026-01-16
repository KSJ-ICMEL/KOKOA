"""
Code Agent - Software Engineer
==============================
Theorist의 코드를 실행하고 컴퓨터공학적 오류를 해결
병렬 디버깅 (3 전략): Direct Fix, Memory Fix, Introspection Fix
"""

import os
import sys
import subprocess
import re
import json
import asyncio
import textwrap
from datetime import datetime
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate

from kokoa.config import Config
from kokoa.state import AgentState, SimulationResult


DEBUG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a **Software Engineer** debugging Python code.

**IMPORTANT:** 
- The Theorist (materials scientist) wrote this code based on physics principles
- Fix ONLY computer science bugs (syntax, imports, API usage)
- Do NOT change the scientific logic or kMC parameters

**Your job:**
1. Analyze the error message
2. Fix the bug
3. Return the corrected code

**Output format:**
```python
# Fixed code here
```

Only output the fixed Python code. No explanations."""),
    ("user", """**Error:**
{error_message}

**Code:**
```python
{code}
```

{additional_context}

Fix the bug and return the corrected code.""")
])


def validate_kmc_code(code: str) -> Tuple[bool, str]:
    forbidden = ["students", "friendships", "infected", "infection", "BFS", "breadth"]
    for pattern in forbidden:
        if pattern.lower() in code.lower():
            return False, f"Invalid code: contains '{pattern}'"
    
    required = ["conductivity"]
    missing = [p for p in required if p.lower() not in code.lower()]
    if missing:
        return False, f"Missing: {missing}"
    
    return True, "OK"


SIMULATION_TIMEOUT = 1800  # 30 minutes


def execute_code(code: str, run_dir: str, iteration: int, timeout: int = SIMULATION_TIMEOUT) -> SimulationResult:
    sim_dir = os.path.join(run_dir, "simulation")
    os.makedirs(sim_dir, exist_ok=True)
    
    script_name = f"{iteration:03d}.py"
    script_path = os.path.abspath(os.path.join(sim_dir, script_name))
    
    indented_code = textwrap.indent(code, '    ')
    safe_run_dir = os.path.abspath(run_dir).replace("\\", "/")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cif_path = os.path.join(project_root, "Li4.47La3Zr2O12.cif").replace("\\", "/")
    safe_project_root = project_root.replace("\\", "/")
    
    wrapped_code = f'''"""
KOKOA Simulation #{iteration}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
import os, sys, traceback

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
        
        return SimulationResult(
            is_success=is_success,
            conductivity=conductivity,
            error_message=stderr if stderr else None,
            execution_log=f"[STDOUT]\n{stdout}\n[STDERR]\n{stderr}"
        )
        
    except subprocess.TimeoutExpired:
        timeout_mins = timeout // 60
        return SimulationResult(
            is_success=False,
            error_message=f"Simulation timed out after {timeout_mins} minutes. Consider simplifying the simulation or reducing parameters.",
            execution_log=f"Execution timed out after {timeout}s ({timeout_mins} min)"
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
    
    result_path = os.path.join(result_dir, f"{iteration:03d}.json")
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "is_success": result.is_success,
        "conductivity": result.conductivity,
        "conductivity_unit": "S/cm",
        "error_message": result.error_message,
        "execution_log": result.execution_log
    }
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _extract_code(response: str) -> str:
    if "```python" in response:
        match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
    if "```" in response:
        match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response.strip()


def _direct_fix(code: str, error: str, llm) -> str:
    """Strategy 1: Direct fix based on error message"""
    prompt_vars = {
        "error_message": error[:1000],
        "code": code[:3000],
        "additional_context": "Fix the error directly based on the error message."
    }
    
    response = ""
    for chunk in llm.stream(DEBUG_PROMPT.format_messages(**prompt_vars)):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        response += content
    
    return _extract_code(response)


def _memory_fix(code: str, error: str, llm, run_dir: str) -> str:
    """Strategy 2: Fix using past successful code patterns"""
    from kokoa.memory import search_memory
    
    skills = search_memory(error[:200], "skills", k=2, run_dir=run_dir)
    
    skills_context = ""
    if skills:
        skills_context = "\n[Reference - Past Successful Code]\n"
        for s in skills:
            skills_context += f"{s['content'][:800]}\n---\n"
    
    prompt_vars = {
        "error_message": error[:1000],
        "code": code[:3000],
        "additional_context": f"Use these successful code patterns as reference:\n{skills_context}" if skills_context else "No past patterns available."
    }
    
    response = ""
    for chunk in llm.stream(DEBUG_PROMPT.format_messages(**prompt_vars)):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        response += content
    
    return _extract_code(response)


def _introspect_fix(code: str, error: str, llm) -> str:
    """Strategy 3: Introspection - analyze imports and API usage with real introspection"""
    from kokoa.tools import quick_introspect, web_search, format_search_results
    
    import_errors = bool(re.search(r"(ImportError|ModuleNotFoundError|AttributeError)", error))
    
    context = ""
    
    if import_errors:
        package_match = re.search(r"No module named '([^']+)'", error)
        attr_match = re.search(r"module '([^']+)' has no attribute '([^']+)'", error)
        
        if package_match:
            pkg = package_match.group(1).split('.')[0]
            try:
                introspect_result = quick_introspect(package_name=pkg)
                if introspect_result.get("classes") or introspect_result.get("functions"):
                    context = f"""[Introspection of {pkg}]
Classes: {introspect_result.get('classes', [])[:3]}
Functions: {introspect_result.get('functions', [])[:5]}"""
            except:
                pass
        
        if attr_match:
            pkg, attr = attr_match.group(1), attr_match.group(2)
            try:
                introspect_result = quick_introspect(package_name=pkg, class_hint=attr)
                if introspect_result.get("classes"):
                    context = f"""[Introspection of {pkg}]
Looking for: {attr}
Available: {introspect_result.get('classes', [])[:3]}"""
            except:
                pass
        
        if not context:
            try:
                search_results = web_search(f"{error[:100]} python fix", max_results=3)
                if search_results:
                    context = f"[Web Search Results]\n{format_search_results(search_results)}"
            except:
                pass
        
        if not context:
            context = """This looks like an import/attribute error.
Common fixes:
- pymatgen.core.structure → from pymatgen.core import Structure
- Check if package is installed
- Verify class/method names"""
    else:
        context = """Analyze the code structure and fix any API usage issues.
- Check function signatures
- Verify return types
- Fix variable scoping issues"""
    
    prompt_vars = {
        "error_message": error[:1000],
        "code": code[:3000],
        "additional_context": context
    }
    
    response = ""
    for chunk in llm.stream(DEBUG_PROMPT.format_messages(**prompt_vars)):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        response += content
    
    return _extract_code(response)


def _parallel_debug(code: str, error: str, llm, run_dir: str) -> List[str]:
    """Run 3 debugging strategies in parallel"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_direct_fix, code, error, llm),
            executor.submit(_memory_fix, code, error, llm, run_dir),
            executor.submit(_introspect_fix, code, error, llm)
        ]
        
        results = []
        for i, future in enumerate(futures):
            try:
                fixed_code = future.result(timeout=60)
                results.append(fixed_code)
            except Exception as e:
                print(f"   Debug strategy {i+1} failed: {e}")
                results.append(None)
        
        return [r for r in results if r]


def _select_best_fix(fixes: List[str], run_dir: str, iteration: int) -> Tuple[Optional[str], Optional[SimulationResult]]:
    """Execute each fix and select the best result"""
    best_code = None
    best_result = None
    
    for i, code in enumerate(fixes):
        if not code:
            continue
        
        is_valid, msg = validate_kmc_code(code)
        if not is_valid:
            print(f"   Fix {i+1} invalid: {msg}")
            continue
        
        result = execute_code(code, run_dir, iteration * 10 + i + 1)
        
        if result.is_success:
            print(f"   Fix {i+1} succeeded! σ = {result.conductivity} S/cm")
            return code, result
        
        if best_result is None or (result.conductivity and (best_result.conductivity is None or abs(result.conductivity - 1.97e-6) < abs(best_result.conductivity - 1.97e-6))):
            best_code = code
            best_result = result
    
    return best_code, best_result


def code_agent_node(state: AgentState, llm) -> dict:
    """Code Agent: Execute code and debug if needed"""
    iteration = state.get("iteration_count", 0) + 1
    code = state.get("python_code", "")
    run_dir = state.get("run_dir")
    research_log = state.get("research_log", [])
    
    print(f"[Code Agent] Iteration {iteration}")
    
    if not code:
        return {
            "simulation_output": SimulationResult(
                is_success=False,
                error_message="No code to execute",
                execution_log="Empty code"
            ),
            "iteration_count": iteration,
            "research_log": research_log + ["CodeAgent: No code"]
        }
    
    is_valid, validation_msg = validate_kmc_code(code)
    if not is_valid:
        print(f"   Validation failed: {validation_msg}")
        last_valid = state.get("last_valid_code", "")
        if last_valid:
            print("   Rolling back to last valid code")
            code = last_valid
        else:
            return {
                "simulation_output": SimulationResult(
                    is_success=False,
                    error_message=f"Validation failed: {validation_msg}",
                    execution_log="Invalid code"
                ),
                "iteration_count": iteration,
                "research_log": research_log + [f"CodeAgent: Validation failed"]
            }
    
    if not run_dir:
        run_dir = os.path.join(Config.RUNS_DIR, "default")
        os.makedirs(run_dir, exist_ok=True)
    
    print("   Executing code...")
    result = execute_code(code, run_dir, iteration)
    
    if result.is_success:
        save_result(result, run_dir, iteration)
        log_msg = f"CodeAgent: Success σ = {result.conductivity} S/cm"
        print(f"   {log_msg}")
        return {
            "python_code": code,
            "simulation_output": result,
            "iteration_count": iteration,
            "last_valid_code": code,
            "research_log": research_log + [log_msg]
        }
    
    print(f"   Execution failed: {result.error_message[:100]}...")
    print("   Starting parallel debugging (3 strategies)...")
    
    fixes = _parallel_debug(code, result.error_message or "", llm, run_dir)
    
    if not fixes:
        print("   All debug strategies failed")
        save_result(result, run_dir, iteration)
        return {
            "python_code": code,
            "simulation_output": result,
            "iteration_count": iteration,
            "research_log": research_log + ["CodeAgent: Debug failed"]
        }
    
    print(f"   Got {len(fixes)} candidate fixes, testing...")
    best_code, best_result = _select_best_fix(fixes, run_dir, iteration)
    
    if best_code and best_result:
        save_result(best_result, run_dir, iteration)
        
        if best_result.is_success:
            log_msg = f"CodeAgent: Fixed! σ = {best_result.conductivity} S/cm"
        else:
            log_msg = f"CodeAgent: Best effort - {best_result.error_message[:50]}"
        
        print(f"   {log_msg}")
        return {
            "python_code": best_code,
            "simulation_output": best_result,
            "iteration_count": iteration,
            "last_valid_code": best_code if best_result.is_success else state.get("last_valid_code", ""),
            "research_log": research_log + [log_msg]
        }
    
    save_result(result, run_dir, iteration)
    return {
        "python_code": code,
        "simulation_output": result,
        "iteration_count": iteration,
        "research_log": research_log + ["CodeAgent: No fix worked"]
    }


def create_code_agent_node(llm):
    def node_fn(state: AgentState) -> dict:
        return code_agent_node(state, llm)
    return node_fn
