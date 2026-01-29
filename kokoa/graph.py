"""
KOKOA Graph Assembly - Single Pass
==================================
Linear Pipeline: Scientist -> Simulator -> Archivist -> END.
"""

import sys
import uuid
import os
from datetime import datetime
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


def build_workflow(scientist_node, simulator_node, archivist_node):
    """
    Build KOKOA Single-Pass Workflow
    Scientist guarantees a new batch of code.
    Simulator runs it.
    Archivist reports and ends.
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Scientist", scientist_node)
    workflow.add_node("Simulator", simulator_node)
    workflow.add_node("Archivist", archivist_node)
    
    # Linear Flow
    workflow.add_edge("Scientist", "Simulator")
    workflow.add_edge("Simulator", "Archivist")
    workflow.add_edge("Archivist", END)
    
    workflow.set_entry_point("Scientist")
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


def run_experiment(app, goal: str, thread_id: str = None):
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    # Recursion limit minimal since it's single pass
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    initial_state = create_initial_state(goal)
    
    run_dir = initial_state.get("run_dir", "unknown")
    run_id = initial_state.get("run_id", thread_id)
    
    output_path = os.path.join(run_dir, "output.txt")
    tee = TeeWriter(output_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    print(f"🚀 KOKOA Start (Run: {run_id}) [Single-Pass Batch]")
    print(f"📁 Output: {run_dir}")
    print(f"🎯 Goal: {goal}")
    print(f"🔧 Model: {Config.MODEL_NAME}")
    print("\n" + "=" * 60 + "\n")
    
    final_state = None
    try:
        for event in app.stream(initial_state, config):
            for node_name, node_output in event.items():
                # Node output is already printed by nodes themselves mostly
                # We can just update final state
                final_state = node_output
        
        print("\n" + "=" * 60)
        print("🏁 Experiment Complete")
        
        if final_state:
            result = final_state.get("simulation_output")
            if result and result.conductivity:
                target = Config.TARGET_CONDUCTIVITY
                import math
                log_error = math.log10(result.conductivity) - math.log10(target)  # Signed
                print(f"📊 Final: σ = {result.conductivity:.2e} S/cm | log(σ) = {math.log10(result.conductivity):.2f} | error = {log_error:+.2f} orders")
        
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
        return None


def save_graph_png(app, output_dir: str = ".") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kokoa_workflow_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    return visualize(app, save_path=filepath)
