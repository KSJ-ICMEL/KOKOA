"""
Researcher Agent
================
arXiv에서 논문을 검색하고 벡터스토어에 임베딩
"""

import os

from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from kokoa.config import Config
from kokoa.state import AgentState


def researcher_node(state: AgentState) -> dict:
    print("📚 [Researcher] arXiv 논문 검색 중...")
    
    query = state.get("research_query") or state.get("goal", "")
    knowledge_gap = state.get("knowledge_gap", "")
    search_query = f"{query} {knowledge_gap}".strip()
    
    research_attempts = state.get("research_attempts", 0) + 1
    research_log = state.get("research_log", [])
    
    try:
        loader = ArxivLoader(
            query=search_query,
            load_max_docs=Config.ARXIV_MAX_DOCS,
            load_all_available_meta=True
        )
        docs = loader.load()
    except Exception as e:
        print(f"   ❌ arXiv 검색 실패: {e}")
        return {
            "needs_research": False,
            "research_attempts": research_attempts,
            "research_log": research_log + [f"Researcher: Search failed - {e}"]
        }
    
    if not docs:
        print("   ❌ 관련 논문을 찾지 못했습니다.")
        return {
            "needs_research": False,
            "research_attempts": research_attempts,
            "research_log": research_log + ["Researcher: No relevant papers found."]
        }
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": Config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Use run-specific chroma_store
    run_dir = state.get("run_dir", ".")
    run_chroma_path = os.path.join(run_dir, "chroma_store")
    
    vectorstore = Chroma(
        persist_directory=run_chroma_path,
        embedding_function=embedding_model
    )
    vectorstore.add_documents(splits)
    
    print(f"   ✅ {len(docs)}개 논문, {len(splits)}개 청크 임베딩 완료")
    for doc in docs:
        title = doc.metadata.get("Title", "Unknown")[:50]
        print(f"      - {title}...")
    
    return {
        "needs_research": False,
        "research_attempts": research_attempts,
        "research_log": research_log + [f"Researcher: Embedded {len(docs)} papers ({len(splits)} chunks)"]
    }
