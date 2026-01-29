"""
Initialize Paper Vector Store from PDFs
========================================
Builds the paper vector store from PDFs in initial_state/pdf/
Used by ll-agent for RAG during scientific analysis.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glob import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from kokoa.config import Config


def init_paper_store(force_rebuild: bool = False):
    """
    Initialize paper vector store from PDFs in initial_state/pdf/
    
    Args:
        force_rebuild: If True, delete existing store and rebuild
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(project_root, "initial_state", "pdf")
    persist_dir = os.path.join(project_root, "initial_state", "pdf_store")
    
    if os.path.exists(persist_dir) and not force_rebuild:
        print(f"✅ Paper store already exists: {persist_dir}")
        print("   Use --force to rebuild")
        return
    
    # Clean existing store if force rebuild
    if os.path.exists(persist_dir) and force_rebuild:
        import shutil
        shutil.rmtree(persist_dir)
        print(f"🗑️ Removed existing store: {persist_dir}")
    
    print(f"🔨 Building paper store from: {pdf_dir}")
    
    # Load PDFs (recursively search subdirectories)
    pdf_files = glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")
    
    documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDF4LLMLoader(pdf_file)
            documents.extend(loader.load())
            print(f"   ✅ Loaded: {os.path.basename(pdf_file)}")
        except Exception as e:
            print(f"   ⚠️ Failed: {os.path.basename(pdf_file)} - {e}")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    print(f"   → Created {len(splits)} chunks")
    
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": Config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Build vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    print(f"✅ Paper store saved: {persist_dir}")
    print(f"   Total documents: {len(documents)}")
    print(f"   Total chunks: {len(splits)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Initialize paper vector store from PDFs")
    parser.add_argument("--force", action="store_true", help="Force rebuild existing store")
    args = parser.parse_args()
    
    init_paper_store(force_rebuild=args.force)
