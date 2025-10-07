# app/services/rag_service.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# LlamaIndex core
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings, Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb

# Embedding & LLM (optional)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter
from dotenv import load_dotenv

class RAGService:
    """
    RAG service using LlamaIndex Auto-Merging Retriever on PDFs under `Alert Guides Docs`.
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    print("✅ OpenRouter key:", (api_key[:5] + "..." + api_key[-5:]) if api_key else "⚠️ Missing")

    def __init__(self, docs_path: str = "Alert Guides Docs"):
        base = Path(__file__).resolve().parents[2]   
        self.docs_dir = (base / docs_path).resolve()
        self._engine: Optional[RetrieverQueryEngine] = None
        self._ready = False
        
        # Debug: print path information
        logger.info(f"RAGService docs_dir: {self.docs_dir}")
        logger.info(f"Docs dir exists: {self.docs_dir.exists()}")

    async def startup(self) -> None:
        """Build index and auto-merging retriever once at app startup."""
        if self._ready:
            return

        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        # Set default LLM/Embedding (Embedding required; LLM optional)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        if api_key:
            Settings.llm = OpenRouter(
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                model="mistralai/mistral-7b-instruct",
                max_tokens=512,
                context_window=4096,
            )
        else:
            Settings.llm = None  # Can still retrieve and return concatenated text without key
            logger.info("No OPENROUTER_API_KEY; LLM generation disabled.")

        # 1) First connect/create Chroma collection (to check if data exists)
        chroma_base = Path(__file__).resolve().parents[2] / "chroma_db"
        chroma_client = chromadb.PersistentClient(path=str(chroma_base))
        chroma_coll = chroma_client.get_or_create_collection("flood_docs")

        vector_store = ChromaVectorStore(chroma_collection=chroma_coll)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

        if chroma_coll.count() and chroma_coll.count() > 0:
            # Use existing data
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            logger.info(f"Using existing Chroma collection with {chroma_coll.count()} vectors.")
        else:
            # Create new index
            documents = []
            
            # Try to load PDF documents
            if self.docs_dir.exists():
                pdfs = list(self.docs_dir.glob("*.pdf"))
                if pdfs:
                    documents = SimpleDirectoryReader(input_files=[str(p) for p in pdfs]).load_data()
                    logger.info(f"Loaded {len(documents)} documents from PDFs")
            
            # If no documents found, create sample documents
            if not documents:
                logger.info("No documents found, creating sample documents")
                self._create_sample_documents()
                documents = [
                    Document(text="Sample flood preparation guide: Always have an emergency kit ready with water, food, and first aid supplies."),
                    Document(text="Government flood response: Emergency services monitor weather conditions and issue warnings when flooding is imminent."),
                    Document(text="Safety protocols: Never walk or drive through flood water. Move to higher ground immediately when advised.")
                ]

            # Split documents → create index → Auto-Merging
            splitter = SentenceSplitter(chunk_size=800, chunk_overlap=80)
            pipeline = IngestionPipeline(transformations=[splitter])
            nodes = pipeline.run(documents=documents)

            index = VectorStoreIndex(nodes, storage_context=storage_ctx)
            logger.info(f"Created new index with {len(nodes)} nodes")

        base_retriever = index.as_retriever(similarity_top_k=12)
        am_retriever = AutoMergingRetriever(base_retriever, index.storage_context)
        self._engine = RetrieverQueryEngine(retriever=am_retriever)
        self._ready = True
        logger.info("RAG Auto-Merging engine is ready.")

    def _create_sample_documents(self):
        """Create sample documents in docs directory"""
        self.docs_dir.mkdir(exist_ok=True)
        sample_content = """
        FLOOD PREPARATION GUIDE
        
        Before a flood:
        1. Create emergency kit with water, food, and medical supplies
        2. Identify evacuation routes
        3. Waterproof important documents
        4. Know emergency contact numbers
        
        During a flood:
        1. Listen to emergency broadcasts
        2. Move to higher ground
        3. Avoid walking or driving through flood water
        
        After a flood:
        1. Wait for official all-clear
        2. Avoid flood water - it may be contaminated
        3. Document damage for insurance
        """
        
        sample_file = self.docs_dir / "sample_flood_guide.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)

    async def get_relevant_content(self, query: str, top_k: int = 6) -> Tuple[str, List[str]]:
        """Run auto-merging retrieval and return (answer, sources[])."""
        if not self._ready or self._engine is None:
            await self.startup()

        q = (query or "What to prepare 7 days before a flood in BC?").strip()

        try:
            resp = self._engine.query(q)
        except Exception as e:
            logger.exception(f"Engine query failed: {e}")
            return f"Query failed: {e}", []

        # 1) Compatible with different return types
        answer = getattr(resp, "response", None)
        if not isinstance(answer, str):
            answer = str(resp)

        # 2) Extract source information (robust: prioritize sn.node.metadata)
        sources: List[str] = []
        seen = set()
        source_nodes = getattr(resp, "source_nodes", None) or []
        for sn in source_nodes:
            # NodeWithScore.node
            node = getattr(sn, "node", None)
            meta = {}
            if node is not None:
                meta = getattr(node, "metadata", None) or {}
            # Fallback: in some implementations metadata might be directly on sn
            if not meta:
                meta = getattr(sn, "metadata", None) or {}

            title = meta.get("file_name") or meta.get("source") or "Unknown"
            if title in seen:
                continue
            seen.add(title)

            # Preview text (try to get from node; fallback to sn.text / node.text)
            preview_text = ""
            try:
                if node is not None and hasattr(node, "get_text"):
                    preview_text = node.get_text()
                elif hasattr(sn, "text") and isinstance(sn.text, str):
                    preview_text = sn.text
                elif node is not None and hasattr(node, "text") and isinstance(node.text, str):
                    preview_text = node.text
            except Exception:
                preview_text = ""

            preview = (preview_text or "").replace("\n", " ")[:200]
            page = meta.get("page_label") or meta.get("page")
            tag = f"{title}{f' (p.{page})' if page else ''} | {preview}..."
            sources.append(tag)

        return answer, sources

     
    def rebuild(self) -> None:
        self._engine = None
        self._ready = False
