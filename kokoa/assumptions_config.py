"""
Assumption Configuration for kMC Simulation
============================================
10 ideal assumptions that can be progressively relaxed toward realistic simulation.
"""

from typing import TypedDict, Optional, List


class AssumptionItem(TypedDict):
    id: str
    category: str
    name: str
    description: str
    status: str
    reason_to_relax: Optional[str]
    implementation_plan: Optional[str]
    relaxed_at_iteration: Optional[int]


BASE_ASSUMPTIONS: List[AssumptionItem] = [
    {
        "id": "A1",
        "category": "Structural",
        "name": "Rigid Lattice Assumption",
        "description": "Host lattice (La, Zr, O) remains fixed. Lattice relaxation or local structural distortion due to lithium migration is not considered.",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A2",
        "category": "Structural",
        "name": "Frozen Framework Assumption",
        "description": "No thermal vibrations (phonons) in the host lattice. Phonon-assisted hopping effects are neglected.",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A3",
        "category": "Structural",
        "name": "Geometric Connectivity Assumption",
        "description": "Lithium migration paths determined solely by geometric proximity (within 4.0 Å). Bottleneck size/shape does not influence migration probability.",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A4",
        "category": "Thermodynamics",
        "name": "Site Energy Equivalence Assumption",
        "description": "All lithium sites (24d, 96h) are energetically equivalent (ΔE_site=0). Jump directionality depends solely on random probability.",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A5",
        "category": "Thermodynamics",
        "name": "Site Exclusion Only Assumption",
        "description": "Li-Li interaction limited to volume exclusion (hard-sphere). Coulombic repulsion effects on activation energy barriers are ignored.",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A6",
        "category": "Thermodynamics",
        "name": "Random Initial State Assumption",
        "description": "Initial distribution of Li ions and vacancies follows completely random distribution, disregarding thermodynamic stability or short-range ordering.",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A7",
        "category": "Kinetics",
        "name": "BKL Algorithm Assumption",
        "description": "Time step (Δt) sampled from exponential distribution inversely proportional to total hopping rate (Bortz-Kalos-Lebowitz algorithm).",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A8",
        "category": "Kinetics",
        "name": "Constant Kinetic Parameters Assumption",
        "description": "Activation energy (E_a=0.30 eV) and attempt frequency (ν=1e13 Hz) are constant for all migration paths, independent of local environment.",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A9",
        "category": "Kinetics",
        "name": "Single Particle Hopping Assumption",
        "description": "Lithium migration modeled as independent single-particle hopping. Multi-ion concerted motion mechanisms are not considered.",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
    {
        "id": "A10",
        "category": "Kinetics",
        "name": "Ideal Nernst-Einstein Assumption",
        "description": "Haven ratio=1. Diffusion is uncorrelated random walk. Tracer diffusion coefficient (D_tracer) equals conductivity diffusion coefficient (D_σ).",
        "status": "active",
        "reason_to_relax": None,
        "implementation_plan": None,
        "relaxed_at_iteration": None,
    },
]


def get_active_assumptions(checklist: List[AssumptionItem]) -> List[AssumptionItem]:
    return [a for a in checklist if a["status"] == "active"]


def get_relaxed_assumptions(checklist: List[AssumptionItem]) -> List[AssumptionItem]:
    return [a for a in checklist if a["status"] == "relaxed"]


def format_assumptions_for_prompt(checklist: List[AssumptionItem]) -> str:
    active = get_active_assumptions(checklist)
    relaxed = get_relaxed_assumptions(checklist)
    
    lines = ["## Active Assumptions (can be relaxed)"]
    for a in active:
        lines.append(f"- [{a['id']}] {a['name']}: {a['description']}")
    
    if relaxed:
        lines.append("\n## Already Relaxed Assumptions")
        for a in relaxed:
            lines.append(f"- [{a['id']}] {a['name']} (iter {a['relaxed_at_iteration']}): {a['reason_to_relax']}")
    
    return "\n".join(lines)


def relax_assumption(
    checklist: List[AssumptionItem],
    assumption_id: str,
    reason: str,
    implementation_plan: str,
    iteration: int
) -> List[AssumptionItem]:
    updated = []
    for a in checklist:
        if a["id"] == assumption_id:
            updated.append({
                **a,
                "status": "relaxed",
                "reason_to_relax": reason,
                "implementation_plan": implementation_plan,
                "relaxed_at_iteration": iteration,
            })
        else:
            updated.append(a)
    return updated
