"""
Archivist Agent - Knowledge Archiving Specialist
=================================================
모든 에이전트로부터 생산된 지식을 수집하고 저장
시뮬레이션 코드 접근이나 종료 판단 없음
"""

from datetime import datetime

from kokoa.config import Config
from kokoa.state import AgentState


def archivist_node(state: AgentState) -> dict:
    """Archivist: Archive all knowledge produced by agents"""
    print("[Archivist] Archiving knowledge from agents...")
    
    result = state.get("simulation_output")
    python_code = state.get("python_code", "")
    hypothesis = state.get("hypothesis", "")
    run_dir = state.get("run_dir")
    iteration = state.get("iteration_count", 0)
    research_log = state.get("research_log", [])
    
    archived_items = []
    
    if hypothesis:
        _save_hypothesis(hypothesis, run_dir, iteration)
        archived_items.append("hypothesis")
    
    if result:
        _save_experiment(state, result, hypothesis, run_dir, iteration)
        archived_items.append("experiment")
        
        if result.is_success and python_code and result.conductivity:
            _save_skill(python_code, hypothesis, result, run_dir, iteration)
            archived_items.append("skill")
        
        if not result.is_success and result.error_message:
            _save_insight(result, hypothesis, run_dir, iteration)
            archived_items.append("insight")
    
    if archived_items:
        print(f"   Archived: {', '.join(archived_items)}")
    else:
        print("   Nothing to archive")
    
    return {
        "research_log": research_log + [f"Archivist: Archived {', '.join(archived_items) or 'nothing'}"]
    }


def _save_hypothesis(hypothesis: str, run_dir: str, iteration: int):
    """Save hypothesis to insights collection"""
    from kokoa.memory import save_to_memory
    
    content = f"""
[Hypothesis - Iteration {iteration}]
{hypothesis}
Timestamp: {datetime.now().isoformat()}
""".strip()
    
    saved = save_to_memory(
        content=content,
        collection="insights",
        metadata={
            "type": "hypothesis",
            "iteration": iteration
        },
        run_dir=run_dir
    )
    
    if saved:
        print(f"   → Saved hypothesis to 'insights'")


def _save_experiment(state, result, hypothesis: str, run_dir: str, iteration: int):
    """Save experiment result to experiments collection"""
    from kokoa.memory import save_to_memory
    
    target = 1.97e-6
    current_val = result.conductivity if result.conductivity else 0.0
    error_rate = abs(target - current_val) / target * 100 if current_val else 100.0
    
    content = f"""
[Experiment - Iteration {iteration}]
Goal: {state.get('goal', 'N/A')[:200]}
Hypothesis: {hypothesis[:300] if hypothesis else 'N/A'}
Success: {result.is_success}
Conductivity: {result.conductivity} S/cm
Error Rate: {error_rate:.2f}%
Timestamp: {datetime.now().isoformat()}
""".strip()
    
    saved = save_to_memory(
        content=content,
        collection="experiments",
        metadata={
            "iteration": iteration,
            "success": result.is_success,
            "conductivity": result.conductivity,
            "error_rate": error_rate
        },
        run_dir=run_dir
    )
    
    if saved:
        print(f"   → Saved experiment to 'experiments'")


def _save_skill(python_code: str, hypothesis: str, result, run_dir: str, iteration: int):
    """Save successful code pattern to skills collection"""
    from kokoa.memory import save_to_memory
    
    target = 1.97e-6
    error_rate = abs(target - result.conductivity) / target * 100 if result.conductivity else 100.0
    
    content = f"""
[Successful Code - Iteration {iteration}]
Conductivity: {result.conductivity} S/cm
Error Rate: {error_rate:.2f}%
Hypothesis: {hypothesis[:200] if hypothesis else 'N/A'}

```python
{python_code[:2500]}
```
""".strip()
    
    saved = save_to_memory(
        content=content,
        collection="skills",
        metadata={
            "iteration": iteration,
            "conductivity": result.conductivity,
            "error_rate": error_rate
        },
        run_dir=run_dir
    )
    
    if saved:
        print(f"   → Saved skill to 'skills'")


def _save_insight(result, hypothesis: str, run_dir: str, iteration: int):
    """Save failure analysis to insights collection"""
    from kokoa.memory import save_to_memory
    
    content = f"""
[Failure Analysis - Iteration {iteration}]
Error: {result.error_message[:500] if result.error_message else 'Unknown'}
Failed Hypothesis: {hypothesis[:200] if hypothesis else 'N/A'}
Timestamp: {datetime.now().isoformat()}
""".strip()
    
    saved = save_to_memory(
        content=content,
        collection="insights",
        metadata={
            "type": "failure",
            "iteration": iteration
        },
        run_dir=run_dir
    )
    
    if saved:
        print(f"   → Saved failure insight to 'insights'")


def create_archivist_node():
    def node_fn(state: AgentState) -> dict:
        return archivist_node(state)
    return node_fn
