"""
Standalone Archivist Runner
============================
Generates a technical report from a completed simulation run.

Usage:
    python tools/run_archivist.py --run-dir runs/20260129_140241

This script:
1. Loads simulation results from run directory
2. Calls the archivist to generate a technical report
3. Saves to vector store + central technical_reports folder
"""

import os
import sys
import json
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from kokoa.config import Config
from kokoa.state import SimulationResult
from kokoa.agents.archivist import archivist_node


def load_simulation_result(run_dir: str) -> SimulationResult:
    """Load simulation result from run directory"""
    result_path = os.path.join(run_dir, "simulation", "initial_state.json")
    
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")
    
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return SimulationResult(
        is_success=data.get("is_success", False),
        conductivity=data.get("conductivity", 0.0),
        diffusivity=data.get("diffusivity", 0.0),
        msd=data.get("msd", 0.0),
        simulation_time_ns=data.get("simulation_time_ns", 0.0),
        temperature_K=data.get("temperature_K", 300),
        steps=data.get("steps", 0),
        error_message=data.get("error_message"),
        execution_log=data.get("execution_log", "")
    )


def load_code(run_dir: str) -> tuple:
    """Load previous and current code from run directory"""
    current_code_path = os.path.join(run_dir, "simulation", "final_state.py")
    previous_code_path = os.path.join(Config.INITIAL_STATE_DIR, "initial_state.py")
    
    current_code = ""
    previous_code = ""
    
    if os.path.exists(current_code_path):
        with open(current_code_path, "r", encoding="utf-8") as f:
            current_code = f.read()
    
    if os.path.exists(previous_code_path):
        with open(previous_code_path, "r", encoding="utf-8") as f:
            previous_code = f.read()
    
    return previous_code, current_code


def main():
    parser = argparse.ArgumentParser(description="Run archivist on a completed simulation")
    parser.add_argument("--run-dir", required=True, help="Path to run directory (e.g., runs/20260129_140241)")
    parser.add_argument("--model", default="gpt-5.1-2025-11-13", help="LLM model to use")
    parser.add_argument("--target", default="", help="Target assumption (e.g., A3)")
    args = parser.parse_args()
    
    # Resolve run directory
    if not os.path.isabs(args.run_dir):
        run_dir = os.path.join(project_root, args.run_dir)
    else:
        run_dir = args.run_dir
    
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    run_id = os.path.basename(run_dir)
    print(f"📦 Loading run: {run_id}")
    
    # Load simulation result
    try:
        result = load_simulation_result(run_dir)
        print(f"   ✅ Simulation result loaded: conductivity={result.conductivity:.4e} S/cm")
    except Exception as e:
        print(f"   ❌ Failed to load result: {e}")
        sys.exit(1)
    
    # Load code
    previous_code, current_code = load_code(run_dir)
    print(f"   ✅ Code loaded: {len(current_code)} bytes")
    
    # Create LLM
    print(f"\n🤖 Initializing LLM: {args.model}")
    Config.MODEL_NAME = args.model  # Set for metadata
    from main import create_llm
    llm = create_llm(args.model)
    
    # Build minimal state for archivist
    state = {
        "run_id": run_id,
        "run_dir": run_dir,
        "simulation_output": result,
        "current_code": current_code,
        "previous_code": previous_code,
        "target_assumption": args.target or f"Manual run {run_id}",
        "relaxed_hurdles": [{"name": args.target or "Manual Analysis", "reason": "Standalone archivist run"}],
        "research_log": []
    }
    
    # Run archivist
    print("\n📝 Running Archivist...")
    archivist_node(state, llm)
    
    print("\n✅ Technical report generated!")


if __name__ == "__main__":
    main()
