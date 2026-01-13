"""
KOKOA Knowledge Base Utilities
==============================
벡터스토어 구축 및 retriever 생성
"""

import os
from glob import glob

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from kokoa.config import Config


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": Config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_knowledge_base(pdf_directory: str = None, force_rebuild: bool = False):
    """
    PDF들로부터 벡터스토어 구축 또는 기존 스토어 로드
    
    Args:
        pdf_directory: PDF 파일 디렉토리 (None이면 Config 사용)
        force_rebuild: True면 기존 스토어 무시하고 재구축
    
    Returns:
        retriever
    """
    pdf_dir = pdf_directory or Config.PDF_DIRECTORY
    persist_dir = Config.PERSIST_DIRECTORY
    
    embedding_model = get_embedding_model()
    
    if os.path.exists(persist_dir) and not force_rebuild:
        print(f"📂 기존 벡터스토어 로드: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    else:
        print(f"🔨 새 벡터스토어 구축...")
        pdf_files = glob(os.path.join(pdf_dir, "*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"PDF 파일 없음: {pdf_dir}")
        
        documents = []
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                documents.extend(loader.load())
                print(f"   ✅ 로드: {os.path.basename(pdf_file)}")
            except Exception as e:
                print(f"   ⚠️ 실패: {os.path.basename(pdf_file)} - {e}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"   → {len(splits)}개 청크 생성")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        print(f"   ✅ 벡터스토어 저장: {persist_dir}")
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Config.K_RETRIEVAL}
    )
    
    return retriever


def get_vectorstore(persist_directory: str = None):
    """Load existing vector store
    
    Args:
        persist_directory: Optional custom path (for run-specific store)
    """
    embedding_model = get_embedding_model()
    persist_dir = persist_directory or Config.PERSIST_DIRECTORY
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
