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
    
    assumptions_checklist = state.get("assumptions_checklist", [])
    current_focus = state.get("current_focus_assumption")
    discovered_gaps = state.get("discovered_gaps", [])
    
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
    
    if current_focus and assumptions_checklist:
        _save_assumption_analysis(assumptions_checklist, current_focus, run_dir, iteration)
        archived_items.append("assumption")
    
    if discovered_gaps:
        _save_discovered_gaps(discovered_gaps, run_dir, iteration)
        archived_items.append("discovered_gaps")
    
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
    
    target = 5.11e-4
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
    
    target = 5.11e-4
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


def _save_assumption_analysis(checklist: list, focus_id: str, run_dir: str, iteration: int):
    """Save assumption analysis to memory"""
    from kokoa.memory import save_to_memory
    
    focused = None
    for a in checklist:
        if a["id"] == focus_id:
            focused = a
            break
    
    if not focused:
        return
    
    content = f"""
[Assumption Analysis - Iteration {iteration}]
ID: {focused['id']}
Name: {focused['name']}
Status: {focused['status']}
Reason to Relax: {focused.get('reason_to_relax', 'N/A')}
Physical Reality: {focused.get('physical_reality', 'N/A')}
Implementation Plan: {focused.get('implementation_plan', 'N/A')}
Timestamp: {datetime.now().isoformat()}
""".strip()
    
    saved = save_to_memory(
        content=content,
        collection="assumption_reviews",
        metadata={
            "assumption_id": focus_id,
            "iteration": iteration,
            "status": focused["status"]
        },
        run_dir=run_dir
    )
    
    if saved:
        print(f"   → Saved assumption [{focus_id}] analysis to 'assumption_reviews'")


def _save_discovered_gaps(gaps: list, run_dir: str, iteration: int):
    """Save newly discovered gaps to insights"""
    from kokoa.memory import save_to_memory
    
    for i, gap in enumerate(gaps):
        content = f"""
[Discovered Gap - Iteration {iteration}]
Gap #{i+1}: {gap}
Timestamp: {datetime.now().isoformat()}
""".strip()
        
        save_to_memory(
            content=content,
            collection="insights",
            metadata={
                "type": "discovered_gap",
                "iteration": iteration
            },
            run_dir=run_dir
        )
    
    print(f"   → Saved {len(gaps)} discovered gap(s) to 'insights'")


def create_archivist_node():
    def node_fn(state: AgentState) -> dict:
        return archivist_node(state)
    return node_fn
