"""
KOKOA Graph Assembly
====================
3-Agent Architecture:
- Scientist: Entry point, knowledge search, code generation, END decision
- CodeAgent: Execution + debugging  
- Archivist: Knowledge archiving (receives from all agents, no routing decisions)

Flow:
  Scientist (entry) → CodeAgent → Archivist → Scientist (loop)
      ↓
     END (when Scientist decides target achieved)
"""

import sys
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from kokoa.config import Config
from kokoa.state import AgentState, create_initial_state


class TeeWriter:
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, data):
        self.stdout.write(data)
        self.stdout.flush()
        self.file.write(data)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def scientist_router(state: AgentState) -> str:
    """Scientist decides: generate code or finish"""
    status = state.get("status", "CONTINUE")
    
    if status == "FINISH":
        print("[Router] Scientist decided: FINISH")
        return "end"
    
    return "code_agent"


def build_workflow(scientist_node, code_agent_node, archivist_node):
    """
    Build KOKOA workflow
    
    Entry: Scientist
    Flow: Scientist → CodeAgent → Archivist → Scientist (loop)
    End: Scientist decides when target is achieved
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Scientist", scientist_node)
    workflow.add_node("CodeAgent", code_agent_node)
    workflow.add_node("Archivist", archivist_node)
    
    workflow.add_conditional_edges(
        "Scientist",
        scientist_router,
        {"code_agent": "CodeAgent", "end": END}
    )
    
    workflow.add_edge("CodeAgent", "Archivist")
    workflow.add_edge("Archivist", "Scientist")
    
    workflow.set_entry_point("Scientist")
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


def run_experiment(app, goal: str, thread_id: str = None):
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    recursion_limit = Config.MAX_LOOPS * 4 + 20
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    initial_state = create_initial_state(goal)
    
    run_dir = initial_state.get("run_dir", "unknown")
    run_id = initial_state.get("run_id", thread_id)
    
    import os
    output_path = os.path.join(run_dir, "output.txt")
    tee = TeeWriter(output_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    print(f"🚀 KOKOA Start (Run: {run_id})")
    print(f"📁 Output: {run_dir}")
    print(f"🎯 Goal: {goal}")
    print(f"🔧 Model: {Config.MODEL_NAME} (memory write: {Config.can_write_memory()})")
    print("\n" + "=" * 60 + "\n")
    
    final_state = None
    try:
        for event in app.stream(initial_state, config):
            for node_name, node_output in event.items():
                print(f"\n{'='*20} [{node_name}] {'='*20}")
                
                if node_name == "Scientist":
                    status = node_output.get('status', 'CONTINUE')
                    hyp = node_output.get('hypothesis', '')[:100]
                    code_len = len(node_output.get('python_code', ''))
                    print(f"Status: {status}")
                    print(f"Hypothesis: {hyp}...")
                    print(f"Code: {code_len} bytes")
                
                elif node_name == "CodeAgent":
                    result = node_output.get("simulation_output")
                    if result:
                        print(f"Success: {result.is_success}")
                        print(f"Conductivity: {result.conductivity} S/cm")
                        if result.error_message and not result.is_success:
                            print(f"Error: {result.error_message[:100]}...")
                
                elif node_name == "Archivist":
                    print("Archived experiment data to memory")
                
                print()
                final_state = node_output
        
        print("\n" + "=" * 60)
        print("🏁 Experiment Complete")
        
        if final_state:
            result = final_state.get("simulation_output") or initial_state.get("simulation_output")
            if result and result.conductivity:
                target = 1.97e-6
                error = abs(target - result.conductivity) / target * 100
                print(f"📊 Final: σ = {result.conductivity} S/cm (error: {error:.1f}%)")
        
    except Exception as e:
        print(f"\n🚨 Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"📝 Log saved: {output_path}")
    
    return final_state


def visualize(app, save_path: str = None):
    try:
        png_data = app.get_graph().draw_mermaid_png()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(png_data)
            print(f"📊 Graph saved: {save_path}")
            return save_path
        else:
            try:
                from IPython.display import Image, display
                display(Image(png_data))
            except ImportError:
                default_path = "workflow_graph.png"
                with open(default_path, "wb") as f:
                    f.write(png_data)
                print(f"📊 Graph saved: {default_path} (IPython not available)")
                return default_path
                
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Requires: pip install grandalf")
        return None


def save_graph_png(app, output_dir: str = ".") -> str:
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kokoa_workflow_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    return visualize(app, save_path=filepath)
