"""
KOKOA Configuration
"""

import os


class Config:
    MODEL_NAME = "deepseek-r1:8b"
    TEMPERATURE = 0.1
    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    K_RETRIEVAL = 3
    
    MAX_LOOPS = 10
    MAX_RESEARCH_ATTEMPTS = 3
    
    PERSIST_DIRECTORY = "./chroma_store"
    PDF_DIRECTORY = "./pdf"
    WORKSPACE_DIR = "./workspace"
    
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 300
    
    ARXIV_MAX_DOCS = 3
    
    @classmethod
    def from_env(cls):
        config = cls()
        config.MODEL_NAME = os.getenv("KOKOA_MODEL", config.MODEL_NAME)
        config.PERSIST_DIRECTORY = os.getenv("KOKOA_CHROMA_DIR", config.PERSIST_DIRECTORY)
        return config
