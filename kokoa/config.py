"""
KOKOA Configuration
"""

import os


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
    MODEL_NAME = "deepseek-r1:8b"
    TEMPERATURE = 0.1
    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_DEVICE = get_device()  # Auto-detect: cuda, mps, or cpu
    K_RETRIEVAL = 3
    
    MAX_LOOPS = 10
    MAX_RESEARCH_ATTEMPTS = 3
    
    INITIAL_STATE_DIR = "./initial_state"
    RUNS_DIR = "./runs"
    
    PERSIST_DIRECTORY = "./chroma_store"
    PDF_DIRECTORY = "./pdf"
    WORKSPACE_DIR = "./workspace"  # Legacy, will be removed
    
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 300
    
    ARXIV_MAX_DOCS = 3
    
    @classmethod
    def from_env(cls):
        config = cls()
        config.MODEL_NAME = os.getenv("KOKOA_MODEL", config.MODEL_NAME)
        config.PERSIST_DIRECTORY = os.getenv("KOKOA_CHROMA_DIR", config.PERSIST_DIRECTORY)
        config.EMBEDDING_DEVICE = os.getenv("KOKOA_DEVICE", config.EMBEDDING_DEVICE)
        return config

