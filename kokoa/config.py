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
    SUPPORTED_MODELS = [
        "gpt-oss:120b",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
    ]
    TEMPERATURE = 0.1
    THINKING_LEVEL = 'high'
    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_DEVICE = get_device()  # Auto-detect: cuda, mps, or cpu
    K_RETRIEVAL = 3
    
    MAX_LOOPS = 10
    MAX_RESEARCH_ATTEMPTS = 3
    
    # Use absolute paths based on project root
    INITIAL_STATE_DIR = os.path.join(_PROJECT_ROOT, "initial_state")
    RUNS_DIR = os.path.join(_PROJECT_ROOT, "runs")
    
    PERSIST_DIRECTORY = os.path.join(_PROJECT_ROOT, "initial_state", "pdf_store")
    PDF_DIRECTORY = os.path.join(_PROJECT_ROOT, "initial_state", "pdf")
    WORKSPACE_DIR = os.path.join(_PROJECT_ROOT, "workspace")  # Legacy
    
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 300
    
    ARXIV_MAX_DOCS = 3
    
    # Simulation parameters
    SIMULATION_TIME = 1000e-9  # Timeout for simulation (1000ns), convergence-based termination is primary
    TARGET_CONDUCTIVITY = 1.97e-6  # Target: experimental LLZO conductivity (S/cm)
    CIF_FILENAME = "LLZO.cif"  # CIF file name in project root
    
    # Timeout settings (unified)
    TIMEOUT = 1800  # 30 minutes for all operations
    
    # Memory permission control
    # Only high-quality models can write to unified memory (prevent hallucination pollution)
    MEMORY_WRITE_ALLOWED_MODELS = [
        "gpt-oss:120b"
    ]
    
    @classmethod
    def can_write_memory(cls) -> bool:
        """Check if current model has permission to write to unified memory"""
        return cls.MODEL_NAME in cls.MEMORY_WRITE_ALLOWED_MODELS
    
    @classmethod
    def from_env(cls):
        config = cls()
        config.MODEL_NAME = os.getenv("KOKOA_MODEL", config.MODEL_NAME)
        config.PERSIST_DIRECTORY = os.getenv("KOKOA_CHROMA_DIR", config.PERSIST_DIRECTORY)
        config.EMBEDDING_DEVICE = os.getenv("KOKOA_DEVICE", config.EMBEDDING_DEVICE)
        return config
    
    @classmethod
    def set_model(cls, model_name: str):
        if model_name in cls.SUPPORTED_MODELS:
            cls.MODEL_NAME = model_name
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose from: {cls.SUPPORTED_MODELS}")


