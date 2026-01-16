#!/usr/bin/env python
"""
KOKOA - Knowledge-Oriented Kinetic Optimization Agent
======================================================
Main execution script

3-Agent Architecture:
- Scientist: Knowledge search + code generation (Theorist + Researcher merged)
- CodeAgent: Execution + parallel debugging
- Archivist: Memory archiving only (no LLM evaluation)

Usage:
    python main.py --goal "Maximize ionic conductivity in LLZO"
    python main.py --interactive
"""

import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_ollama import ChatOllama

from kokoa.config import Config
from kokoa.knowledge import build_knowledge_base
from kokoa.agents.scientist import create_scientist_node
from kokoa.agents.code_agent import create_code_agent_node
from kokoa.agents.archivist import create_archivist_node
from kokoa.graph import build_workflow, run_experiment, visualize


def main():
    parser = argparse.ArgumentParser(description="KOKOA Agent System")
    parser.add_argument("--goal", type=str, help="Research goal")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--rebuild-kb", action="store_true", help="Force rebuild knowledge base")
    parser.add_argument("--visualize", action="store_true", help="Visualize graph structure")
    args = parser.parse_args()
    
    print("🔧 Initializing KOKOA...")
    print(f"   Model: {Config.MODEL_NAME}")
    print(f"   Memory Write: {Config.can_write_memory()}")
    
    llm = ChatOllama(
        model=Config.MODEL_NAME,
        temperature=Config.TEMPERATURE
    )
    print(f"   ✅ LLM ready")
    
    retriever = build_knowledge_base(force_rebuild=args.rebuild_kb)
    print(f"   ✅ Knowledge Base ready")
    
    scientist = create_scientist_node(retriever, llm)
    code_agent = create_code_agent_node(llm)
    archivist = create_archivist_node()
    
    app = build_workflow(scientist, code_agent, archivist)
    print("   ✅ Workflow built")
    print(f"       Scientist → CodeAgent → Archivist")
    
    if args.visualize:
        from kokoa.graph import save_graph_png
        save_graph_png(app, ".")
        return
    
    if args.interactive:
        print("\n" + "=" * 60)
        print("KOKOA Interactive Mode")
        print("Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            goal = input("\n🎯 Research Goal: ").strip()
            if goal.lower() == 'quit':
                break
            if not goal:
                continue
            run_experiment(app, goal)
    
    elif args.goal:
        run_experiment(app, args.goal)
    
    else:
        default_goal = """
        Objective: Optimize the ionic conductivity of a 3D lattice model representing a solid electrolyte.
        Target: Predict conductivity = 1.97e-6 S/cm.
        """.strip()
        
        print(f"\nUsing default goal:")
        print(default_goal)
        
        run_experiment(app, default_goal)


if __name__ == "__main__":
    main()
