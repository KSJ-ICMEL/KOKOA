#!/usr/bin/env python
"""
KOKOA - Knowledge-Oriented Kinetic Optimization Agent
======================================================
메인 실행 스크립트

Usage:
    python main.py --goal "Maximize ionic conductivity in LLZO"
    python main.py --interactive
"""

import argparse

from langchain_ollama import ChatOllama

from kokoa.config import Config
from kokoa.knowledge import build_knowledge_base
from kokoa.agents.theorist import create_theorist_node
from kokoa.agents.engineer import create_engineer_node
from kokoa.agents.simulator import simulator_node
from kokoa.agents.analyst import create_analyst_node
from kokoa.graph import build_workflow, run_experiment, visualize


def main():
    parser = argparse.ArgumentParser(description="KOKOA Agent System")
    parser.add_argument("--goal", type=str, help="Research goal")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--rebuild-kb", action="store_true", help="Force rebuild knowledge base")
    parser.add_argument("--visualize", action="store_true", help="Visualize graph structure")
    args = parser.parse_args()
    
    print("🔧 KOKOA 초기화 중...")
    
    llm = ChatOllama(
        model=Config.MODEL_NAME,
        temperature=Config.TEMPERATURE
    )
    print(f"   ✅ LLM: {Config.MODEL_NAME}")
    
    retriever = build_knowledge_base(force_rebuild=args.rebuild_kb)
    print(f"   ✅ Knowledge Base 준비 완료")
    
    theorist = create_theorist_node(retriever, llm)
    engineer = create_engineer_node(llm)
    analyst = create_analyst_node(llm)
    
    app = build_workflow(theorist, engineer, simulator_node, analyst)
    print("   ✅ 워크플로우 빌드 완료")
    
    if args.visualize:
        visualize(app)
        return
    
    if args.interactive:
        print("\n" + "=" * 60)
        print("KOKOA Interactive Mode")
        print("Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            goal = input("\n🎯 연구 목표 입력: ").strip()
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
        
        print("\n기본 목표 사용:")
        print(default_goal)
        
        run_experiment(app, default_goal)


if __name__ == "__main__":
    main()
