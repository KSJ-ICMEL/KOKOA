"""
KOKOA - Knowledge-Oriented Kinetic Optimization Agent
======================================================
Multi-Agent System for Solid Electrolyte Optimization

3-Agent Architecture:
- Scientist: Knowledge Search, Hypothesis, Code Gen, Strategy
- Simulator: Execution
- Archivist: Archiving & Analysis
"""

from kokoa.config import Config
from kokoa.state import AgentState, SimulationResult, create_initial_state
from kokoa.graph import build_workflow, run_experiment, visualize, save_graph_png
from kokoa.agents import (
    scientist_node, create_scientist_node,
    create_simulator_node,
    archivist_node, create_archivist_node,
)

__version__ = "0.2.1"
__all__ = [
    # Config
    "Config",
    # State
    "AgentState",
    "SimulationResult",
    "create_initial_state",
    # Graph
    "build_workflow",
    "run_experiment",
    "visualize",
    "save_graph_png",
    # Agents
    "scientist_node",
    "create_scientist_node",
    "create_simulator_node",
    "archivist_node",
    "create_archivist_node",
]
