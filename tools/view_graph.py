#!/usr/bin/env python
"""
KOKOA Graph Visualization
=========================
워크플로우 그래프만 저장 (LLM/KB 로딩 없이 빠르게 실행)

Usage:
    python save_graph.py
    python save_graph.py --output docs/workflow.png
"""

import argparse
import os
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from kokoa.state import AgentState


def build_graph_structure():
    """Build workflow graph structure only (no actual agent nodes)"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Scientist", lambda x: x)
    workflow.add_node("CodeAgent", lambda x: x)
    workflow.add_node("Archivist", lambda x: x)
    
    def scientist_router(state):
        return "code_agent" if state.get("status") != "FINISH" else "end"
    
    workflow.add_conditional_edges(
        "Scientist",
        scientist_router,
        {"code_agent": "CodeAgent", "end": END}
    )
    
    workflow.add_edge("CodeAgent", "Archivist")
    workflow.add_edge("Archivist", "Scientist")
    
    workflow.set_entry_point("Scientist")
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def save_graph(output_path: str = None) -> str:
    """Save workflow graph as PNG"""
    app = build_graph_structure()
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"kokoa_workflow_{timestamp}.png"
    
    try:
        png_data = app.get_graph().draw_mermaid_png()
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(png_data)
        
        print(f"📊 Graph saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("   Requires: pip install grandalf")
        return None


def main():
    parser = argparse.ArgumentParser(description="Save KOKOA workflow graph as PNG")
    parser.add_argument("--output", "-o", type=str, help="Output file path (default: auto-generated)")
    args = parser.parse_args()
    
    print("🔧 KOKOA Graph Visualization")
    print("   (No LLM/KB loading - fast execution)")
    print()
    
    save_graph(args.output)


if __name__ == "__main__":
    main()
