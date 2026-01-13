"""
KOKOA Configuration
"""

import os

# Project root = directory containing kokoa/ (one level up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    """Auto-detect best available device (cuda > mps > cpu)"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except ImportError:
        pass
    return "cpu"


class Config:
    MODEL_NAME = "gpt-oss:120b"
    TEMPERATURE = 0.1
    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_DEVICE = get_device()  # Auto-detect: cuda, mps, or cpu
    K_RETRIEVAL = 3
    
    MAX_LOOPS = 10
    MAX_RESEARCH_ATTEMPTS = 3
    
    # Use absolute paths based on project root
    INITIAL_STATE_DIR = os.path.join(_PROJECT_ROOT, "initial_state")
    RUNS_DIR = os.path.join(_PROJECT_ROOT, "runs")
    
    PERSIST_DIRECTORY = os.path.join(_PROJECT_ROOT, "initial_state", "chroma_store")
    PDF_DIRECTORY = os.path.join(_PROJECT_ROOT, "initial_state", "pdf")
    WORKSPACE_DIR = os.path.join(_PROJECT_ROOT, "workspace")  # Legacy
    
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 300
    
    ARXIV_MAX_DOCS = 3
    
    # Simulation parameters
    SIMULATION_TIME = 5e-9  # Target simulation time in seconds (default: 5ns)
    
    @classmethod
    def from_env(cls):
        config = cls()
        config.MODEL_NAME = os.getenv("KOKOA_MODEL", config.MODEL_NAME)
        config.PERSIST_DIRECTORY = os.getenv("KOKOA_CHROMA_DIR", config.PERSIST_DIRECTORY)
        config.EMBEDDING_DEVICE = os.getenv("KOKOA_DEVICE", config.EMBEDDING_DEVICE)
        return config

