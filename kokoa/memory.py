"""
KOKOA Unified Memory System
===========================
Simple is Best: 최소한의 컬렉션으로 최대 효율

Collections:
- papers: 외부 논문 (RAG 검색용)
- technical_reports: LLM 분석 기술 보고서 (result_type으로 필터링)
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from kokoa.config import Config


_COLLECTIONS = ["papers", "technical_reports"]
_embedding_model = None
_vector_stores: Dict[str, Chroma] = {}


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": Config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_model


def _get_vector_store(collection: str, run_dir: str = None) -> Chroma:
    if collection not in _COLLECTIONS:
        raise ValueError(f"Unknown collection: {collection}. Must be one of {_COLLECTIONS}")
    
    # Simple is Best: no nested 'memory' folder
    if run_dir:
        persist_dir = os.path.join(run_dir, collection)
    else:
        # technical_reports -> technical_report_store for consistency
        folder_name = "technical_report_store" if collection == "technical_reports" else collection
        persist_dir = os.path.join(Config.INITIAL_STATE_DIR, folder_name)
    
    os.makedirs(persist_dir, exist_ok=True)
    
    cache_key = f"{persist_dir}_{collection}"
    if cache_key not in _vector_stores:
        _vector_stores[cache_key] = Chroma(
            collection_name=collection,
            embedding_function=_get_embedding_model(),
            persist_directory=persist_dir
        )
    
    return _vector_stores[cache_key]


def save_to_memory(
    content: str,
    collection: str,
    metadata: Dict[str, Any] = None,
    run_dir: str = None,
    force: bool = False
) -> bool:
    """
    Save content to unified memory
    
    Args:
        content: Text content to save
        collection: One of 'papers', 'technical_reports'
        metadata: Optional metadata dict
        run_dir: Run-specific directory (None for global memory)
        force: Bypass permission check (use with caution)
    
    Returns:
        True if saved, False if permission denied
    """
    if not force and not Config.can_write_memory():
        print(f"   [Memory] Write denied: {Config.MODEL_NAME} not in allowed models")
        return False
    
    if metadata is None:
        metadata = {}
    
    metadata["timestamp"] = datetime.now().isoformat()
    metadata["model"] = Config.MODEL_NAME
    
    store = _get_vector_store(collection, run_dir)
    store.add_texts(
        texts=[content],
        metadatas=[metadata]
    )
    
    print(f"   [Memory] Saved to '{collection}': {content[:50]}...")
    return True


def search_memory(
    query: str,
    collection: str,
    k: int = 3,
    run_dir: str = None,
    include_global: bool = True,
    filter: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Search unified memory
    
    Args:
        query: Search query
        collection: One of 'papers', 'technical_reports'
        k: Number of results to return
        run_dir: Run-specific directory
        include_global: Also search global memory (initial_state/memory/)
        filter: Metadata filter (e.g. {"result_type": "FAILURE"})
    
    Returns:
        List of results with 'content' and 'metadata' keys
    """
    results = []
    
    if run_dir:
        try:
            store = _get_vector_store(collection, run_dir)
            docs = store.similarity_search(query, k=k, filter=filter)
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": "run"
                })
        except Exception:
            pass
    
    if include_global:
        try:
            global_store = _get_vector_store(collection, run_dir=None)
            docs = global_store.similarity_search(query, k=k, filter=filter)
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": "global"
                })
        except Exception:
            pass
    
    seen = set()
    unique_results = []
    for r in results:
        key = r["content"][:2000]
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    return unique_results[:k]


def get_memory_stats(run_dir: str = None) -> Dict[str, int]:
    """Get count of items in each collection"""
    stats = {}
    for collection in _COLLECTIONS:
        try:
            store = _get_vector_store(collection, run_dir)
            stats[collection] = store._collection.count()
        except Exception:
            stats[collection] = 0
    return stats


def format_memory_context(results: List[Dict[str, Any]], prefix: str = "") -> str:
    """Format memory search results for LLM context"""
    if not results:
        return f"{prefix}No relevant memory found."
    
    lines = [f"{prefix}[Memory Search Results]"]
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        content = r["content"]
        lines.append(f"{prefix}[{i}] ({source}) {content}")
    
    return "\n".join(lines)
