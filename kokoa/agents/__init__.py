"""
KOKOA Agents Package
====================
3-Agent Architecture:
- Scientist: Entry point, knowledge search, code generation, END decision
- CodeAgent: Execution + debugging
- Archivist: Knowledge archiving
"""

from kokoa.agents.scientist import scientist_node, create_scientist_node
from kokoa.agents.code_agent import code_agent_node, create_code_agent_node
from kokoa.agents.archivist import archivist_node, create_archivist_node

__all__ = [
    # Scientist (Theorist + Researcher merged)
    "scientist_node",
    "create_scientist_node",
    # Code Agent (Engineer + Simulator merged)
    "code_agent_node",
    "create_code_agent_node",
    # Archivist (Analyst merged)
    "archivist_node",
    "create_archivist_node",
]
