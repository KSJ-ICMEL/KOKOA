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
from kokoa.agents.simulator import create_simulator_node
from kokoa.agents.archivist import create_archivist_node
from kokoa.graph import build_workflow, run_experiment, visualize


def create_llm(model_name: str):
    """Create LLM based on model name"""
    GEMINI_MODELS = ["gemini-2.5-pro", "gemini-3-flash-preview"]
    OPENAI_MODELS = ["gpt-5.1-2025-11-13"]
    
    if model_name in GEMINI_MODELS:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=Config.TEMPERATURE,
        )
    elif model_name in OPENAI_MODELS:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=Config.TEMPERATURE,
        )
    else:
        # Ollama models (gpt-oss:120b, llama, etc.)
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            temperature=Config.TEMPERATURE,
        )


def main():
    parser = argparse.ArgumentParser(description="KOKOA Agent System")
    parser.add_argument("-m", "--model", type=str, 
                        choices=Config.SUPPORTED_MODELS,
                        required=True,
                        help=f"Model to use (required). Choices: {Config.SUPPORTED_MODELS}")
    parser.add_argument("-p", "--prompt", type=str,
                        help="Specific instruction for the scientist (e.g., 'Relax assumption A1')")
    args = parser.parse_args()
    
    Config.set_model(args.model)
    
    print("Initializing KOKOA...")
    print(f"   Model: {Config.MODEL_NAME}")
    
    llm = create_llm(Config.MODEL_NAME)
    print(f"   LLM ready")
    
    retriever = build_knowledge_base()
    print(f"   ✅ Knowledge Base ready")
    
    scientist = create_scientist_node(retriever, llm)
    simulator = create_simulator_node()
    archivist = create_archivist_node(llm)  # LLM-powered analysis
    
    app = build_workflow(scientist, simulator, archivist)
    print("   ✅ Workflow built")
    print(f"       Scientist → Simulator → Archivist")
    
    # Build goal from prompt or use default
    if args.prompt:
        goal = f"""
        Objective: {args.prompt}
        
        Context: You are modifying a kMC simulation for Li-ion conductivity in LLZO.
        Validation: Experimental bulk Li-ion conductivity = 1.63e-6 S/cm at 300K.
        Focus on the specific instruction above and generate ONLY the code changes needed.
        """.strip()
        print(f"\n🎯 User Instruction: {args.prompt}")
    else:
        goal = """
        Objective: Reduce the gap between simulation and reality by progressively relaxing idealized assumptions in the kMC model.
        Approach: Add realistic physical factors (site energies, correlation effects, lattice dynamics) and governing equations to capture real ionic transport behavior.
        Validation: Experimental bulk Li-ion conductivity = 1.63e-6 S/cm at 300K. The simulation value should naturally approach this target as physical realism improves.
        """.strip()
        print(f"\nUsing default goal")
    
    run_experiment(app, goal)


if __name__ == "__main__":
    main()
