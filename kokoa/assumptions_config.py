from typing import List, TypedDict, Optional

class AssumptionItem(TypedDict):
    id: str
    category: str
    name: str
    description: str

BASE_ASSUMPTIONS: List[AssumptionItem] = [
    {
        "id": "A1",
        "category": "Structural",
        "name": "Rigid Lattice Assumption",
        "description": "Host lattice (La, Zr, O) remains fixed. Lattice relaxation or local structural distortion due to lithium migration is not considered.",
    },
    {
        "id": "A2",
        "category": "Structural",
        "name": "Frozen Framework Assumption",
        "description": "No thermal vibrations (phonons) in the host lattice. Phonon-assisted hopping effects are neglected.",
    },
    {
        "id": "A3",
        "category": "Structural",
        "name": "Geometric Connectivity Assumption",
        "description": "Lithium migration paths determined solely by geometric proximity (within 4.0 Å). Bottleneck size/shape does not influence migration probability.",
    },
    {
        "id": "A4",
        "category": "Thermodynamics",
        "name": "Site Energy Equivalence Assumption",
        "description": "All lithium sites (24d, 96h) are energetically equivalent (ΔE_site=0). Jump directionality depends solely on random probability.",
    },
    {
        "id": "A5",
        "category": "Thermodynamics",
        "name": "Site Exclusion Only Assumption",
        "description": "Li-Li interaction limited to volume exclusion (hard-sphere). Coulombic repulsion effects on activation energy barriers are ignored.",
    },
    {
        "id": "A6",
        "category": "Thermodynamics",
        "name": "Random Initial State Assumption",
        "description": "Initial distribution of Li ions and vacancies follows completely random distribution, disregarding thermodynamic stability or short-range ordering.",
    },
    {
        "id": "A7",
        "category": "Kinetics",
        "name": "BKL Algorithm Assumption",
        "description": "Time step (Δt) sampled from exponential distribution inversely proportional to total hopping rate (Bortz-Kalos-Lebowitz algorithm).",
    },
    {
        "id": "A8",
        "category": "Kinetics",
        "name": "Constant Kinetic Parameters Assumption",
        "description": "Activation energy (E_a=0.30 eV) and attempt frequency (ν=1e13 Hz) are constant for all migration paths, independent of local environment.",
    },
    {
        "id": "A9",
        "category": "Kinetics",
        "name": "Single Particle Hopping Assumption",
        "description": "Lithium migration modeled as independent single-particle hopping. Multi-ion concerted motion mechanisms are not considered.",
    },
    {
        "id": "A10",
        "category": "Kinetics",
        "name": "Ideal Nernst-Einstein Assumption",
        "description": "Haven ratio=1. Diffusion is uncorrelated random walk. Tracer diffusion coefficient (D_tracer) equals conductivity diffusion coefficient (D_σ).",
    },
]

def format_assumptions_for_prompt(assumptions: List[AssumptionItem]) -> str:
    """Format assumptions for LLM prompt"""
    lines = []
    lines.append("## Simulation Assumptions List")
    for a in assumptions:
        lines.append(f"- [{a['id']}] {a['name']}: {a['description']}")
    return "\n".join(lines)
