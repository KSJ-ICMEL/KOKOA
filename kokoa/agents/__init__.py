"""
KOKOA Agents Package
====================
3-Agent Architecture:
- Scientist: Entry point, knowledge search, code generation, END decision
- Simulator: Code execution (simple)
- Archivist: Knowledge archiving
"""

from kokoa.agents.scientist import scientist_node, create_scientist_node
from kokoa.agents.simulator import create_simulator_node
from kokoa.agents.archivist import archivist_node, create_archivist_node

__all__ = [
    # Scientist
    "scientist_node",
    "create_scientist_node",
    # Simulator
    "create_simulator_node",
    # Archivist
    "archivist_node",
    "create_archivist_node",
]
