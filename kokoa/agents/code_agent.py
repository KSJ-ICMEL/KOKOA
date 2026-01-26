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

from langchain_core.runnables import RunnableConfig

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
3. Return a summary of the fix and the corrected code

**Output format:**
```plain text
[Error Analysis]: Brief explanation of the error
[Fix]: How you fixed it
```

```python
# Fixed code here
```

Output ONLY the two blocks above."""),
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


SIMULATION_TIMEOUT = Config.TIMEOUT  # Use unified timeout from config


def _cleanup_scientist_code(code: str) -> str:
    """Remove redundant code that is already provided by the wrapper.
    
    The wrapper provides: os, sys, json, numpy (as np), Structure, structure (with supercell)
    """
    if not code:
        return code
    
    lines = code.split('\n')
    cleaned_lines = []
    
    # Patterns to remove (these are already in wrapper)
    skip_patterns = [
        'import numpy',
        'import np',
        'from numpy import',
        'import os',
        'from os import',
        'import sys',
        'from sys import',
        'import json',
        'from json import',
        'from pymatgen.core.structure import Structure',
        'from pymatgen.core import Structure',
        'from pymatgen import Structure',
        'import traceback',
    ]
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines at the very start
        if not cleaned_lines and not stripped:
            continue
        
        # Check if this line should be skipped
        should_skip = False
        for pattern in skip_patterns:
            if stripped.startswith(pattern) or pattern in stripped and ('import' in stripped or 'from' in stripped):
                should_skip = True
                break
        
        # Skip Structure.from_file lines (structure is pre-loaded)
        if 'Structure.from_file' in stripped:
            should_skip = True
        
        if not should_skip:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def execute_code(code: str, run_dir: str, iteration: int, timeout: int = SIMULATION_TIMEOUT) -> SimulationResult:
    sim_dir = os.path.join(run_dir, "simulation")
    os.makedirs(sim_dir, exist_ok=True)
    
    script_name = f"{datetime.now().strftime('%y%m%d%H%M%S')}.py"
    script_path = os.path.abspath(os.path.join(sim_dir, script_name))
    
    # Copy CIF file to simulation directory (Essential for self-contained execution)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_cif = os.path.join(project_root, "initial_state", Config.CIF_FILENAME)
    local_cif = os.path.join(sim_dir, Config.CIF_FILENAME)
    
    if not os.path.exists(local_cif) and os.path.exists(source_cif):
        import shutil
        shutil.copy(source_cif, local_cif)

    # Write code directly (Scientist provides full script)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    try:
        # Use Popen for real-time streaming output
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=sim_dir  # Execute inside simulation dir so LLZO.cif is found
        )
        
        stdout_lines = []
        stderr_lines = []
        
        print("\n--- Simulation Output ---")
        
        # Stream stdout in real-time
        import threading
        
        def read_stderr():
            for line in process.stderr:
                stderr_lines.append(line)
        
        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.start()
        
        # Read stdout line by line and print immediately
        for line in process.stdout:
            print(line, end='', flush=True)
            stdout_lines.append(line)
        
        # Wait for process to complete with timeout
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            raise
        
        stderr_thread.join(timeout=5)
        
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)
        is_success = process.returncode == 0
        
        if stderr and not is_success:
            print(f"--- Errors ---\n{stderr}")
        
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


def _extract_fix_info(response: str) -> Tuple[str, str]:
    """Extract (fixed_code, debug_summary) from LLM response"""
    debug_summary = "No debug summary provided."
    fixed_code = response.strip()

    # Extract plain text summary
    summary_match = re.search(r'```plain text\s*(.*?)\s*```', response, re.DOTALL)
    if summary_match:
        debug_summary = summary_match.group(1).strip()
    
    # Extract python code
    if "```python" in response:
        match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            fixed_code = match.group(1).strip()
    elif "```" in response:
        # Fallback for generic block if python one missing
        match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if match:
             # If the first block was plain text, find the second one
            matches = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if len(matches) > 1 and summary_match:
                fixed_code = matches[1].strip()
            else:
                fixed_code = match.group(1).strip()
                
    return fixed_code, debug_summary


def _direct_fix(code: str, error: str, llm, config: RunnableConfig) -> Tuple[str, str]:
    """Strategy 1: Direct fix based on error message"""
    prompt_vars = {
        "error_message": error,
        "code": code,
        "additional_context": "Fix the error directly based on the error message."
    }
    
    response = ""
    for chunk in llm.stream(DEBUG_PROMPT.format_messages(**prompt_vars), config=config):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        response += content
    
    return _extract_fix_info(response)


def _memory_fix(code: str, error: str, llm, run_dir: str, config: RunnableConfig) -> Tuple[str, str]:
    """Strategy 2: Fix using past successful code patterns"""
    from kokoa.memory import search_memory
    
    skills = search_memory(error, "skills", k=2, run_dir=run_dir)
    
    skills_context = ""
    if skills:
        skills_context = "\n[Reference - Past Successful Code]\n"
        for s in skills:
            skills_context += f"{s['content'][:800]}\n---\n"
    
    prompt_vars = {
        "error_message": error,
        "code": code,
        "additional_context": f"Use these successful code patterns as reference:\n{skills_context}" if skills_context else "No past patterns available."
    }
    
    response = ""
    response = ""
    for chunk in llm.stream(DEBUG_PROMPT.format_messages(**prompt_vars), config=config):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        response += content
    
    return _extract_fix_info(response)


def _introspect_fix(code: str, error: str, llm, config: RunnableConfig) -> Tuple[str, str]:
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
        "error_message": error,
        "code": code,
        "additional_context": context
    }
    
    response = ""
    response = ""
    for chunk in llm.stream(DEBUG_PROMPT.format_messages(**prompt_vars), config=config):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        response += content
    
    return _extract_fix_info(response)


def _parallel_debug(code: str, error: str, llm, run_dir: str, config: RunnableConfig) -> List[Tuple[str, str]]:
    """Run 3 debugging strategies in parallel. Returns list of (code, summary)"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_direct_fix, code, error, llm, config),
            executor.submit(_memory_fix, code, error, llm, run_dir, config),
            executor.submit(_introspect_fix, code, error, llm, config)
        ]
        
        results = []
        for i, future in enumerate(futures):
            try:
                fixed_code, summary = future.result(timeout=Config.TIMEOUT)
                results.append((fixed_code, summary))
            except Exception as e:
                print(f"   Debug strategy {i+1} failed: {e}")
                results.append(None)
        
        return [r for r in results if r]


def _select_best_fix(fixes: List[str], run_dir: str, iteration: int) -> Tuple[Optional[str], Optional[SimulationResult]]:
    """Execute each fix and select the best result"""
    best_code = None
    best_result = None
    
def _select_best_fix(fixes: List[Tuple[str, str]], run_dir: str, iteration: int) -> Tuple[Optional[str], Optional[SimulationResult], str]:
    """Execute each fix and select the best result. Returns (code, result, summary)"""
    best_code = None
    best_result = None
    best_summary = ""
    
    for i, (code, summary) in enumerate(fixes):
        if not code:
            continue
        
        is_valid, msg = validate_kmc_code(code)
        if not is_valid:
            print(f"   Fix {i+1} invalid: {msg}")
            continue
        
        result = execute_code(code, run_dir, iteration * 10 + i + 1)
        
        if result.is_success:
            print(f"   Fix {i+1} succeeded! σ = {result.conductivity} S/cm")
            return code, result, summary
        
        if best_result is None or (result.conductivity and (best_result.conductivity is None or abs(result.conductivity - Config.TARGET_CONDUCTIVITY) < abs(best_result.conductivity - Config.TARGET_CONDUCTIVITY))):
            best_code = code
            best_result = result
            best_summary = summary
    
    return best_code, best_result, best_summary


def code_agent_node(state: AgentState, llm, config: RunnableConfig) -> dict:
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
            "scientist_code": code, # Save original Scientist code
            "simulation_output": result,
            "iteration_count": iteration,
            "last_valid_code": code,
            "research_log": research_log + [log_msg]
        }
    
    print(f"   Execution failed: {result.error_message[:100]}...")
    print("   Starting parallel debugging (3 strategies)...")
    
    fixes = _parallel_debug(code, result.error_message or "", llm, run_dir, config)
    
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
    best_code, best_result, best_summary = _select_best_fix(fixes, run_dir, iteration)
    
    if best_code and best_result:
        save_result(best_result, run_dir, iteration)
        
        if best_result.is_success:
            log_msg = f"CodeAgent: Fixed! σ = {best_result.conductivity} S/cm"
        else:
            log_msg = f"CodeAgent: Best effort - {best_result.error_message[:50]}"
        
        print(f"   {log_msg}")
        return {
            "python_code": best_code,
            "scientist_code": code, # Preserve original broken code for diff
            "debug_summary": best_summary,
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
    def node_fn(state: AgentState, config: RunnableConfig) -> dict:
        return code_agent_node(state, llm, config)
    return node_fn
