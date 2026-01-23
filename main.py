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
    python main.py -m gemini-2.5-pro --goal "..."
"""

import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from kokoa.config import Config
from kokoa.knowledge import build_knowledge_base
from kokoa.agents.scientist import create_scientist_node
from kokoa.agents.code_agent import create_code_agent_node
from kokoa.agents.archivist import create_archivist_node
from kokoa.graph import build_workflow, run_experiment, visualize


def create_llm(model_name: str):
    """Create LLM based on model name"""
    if model_name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=Config.TEMPERATURE
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            temperature=Config.TEMPERATURE
        )


def main():
    parser = argparse.ArgumentParser(description="KOKOA Agent System")
    parser.add_argument("-m", "--model", type=str, 
                        choices=Config.SUPPORTED_MODELS,
                        required=True,
                        help=f"Model to use (required). Choices: {Config.SUPPORTED_MODELS}")
    parser.add_argument("--goal", type=str, help="Research goal")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--rebuild-kb", action="store_true", help="Force rebuild knowledge base")
    parser.add_argument("--visualize", action="store_true", help="Visualize graph structure")
    args = parser.parse_args()
    
    Config.set_model(args.model)
    
    print("Initializing KOKOA...")
    print(f"   Model: {Config.MODEL_NAME}")
    print(f"   Memory Write: {Config.can_write_memory()}")
    
    llm = create_llm(Config.MODEL_NAME)
    print(f"   LLM ready")
    
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
        Objective: Reduce the gap between simulation and reality by progressively relaxing idealized assumptions in the kMC model.
        Approach: Add realistic physical factors (site energies, correlation effects, lattice dynamics) and governing equations to capture real ionic transport behavior.
        Validation: Experimental bulk Li-ion conductivity = 1.63e-6 S/cm. The simulation value should naturally approach this target as physical realism improves.
        """.strip()
        
        print(f"\nUsing default goal:")
        print(default_goal)
        
        run_experiment(app, default_goal)


if __name__ == "__main__":
    main()
