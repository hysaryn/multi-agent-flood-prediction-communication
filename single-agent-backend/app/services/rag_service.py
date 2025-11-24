import os
import json
from typing import List, Tuple, Optional
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service for flood prediction and emergency response.
    This service handles document ingestion, vector storage, and semantic search.
    """
    
    def __init__(self, docs_path: str = "docs", collection_name: str = "flood_docs"):
        self.docs_path = Path(docs_path)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self._get_or_create_collection()
        
        # Initialize documents if collection is empty
        if self.collection.count() == 0:
            self._ingest_documents()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            return self.client.get_collection(name=self.collection_name)
        except:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Flood prediction and emergency response documents"}
            )
    
    def _ingest_documents(self):
        """Ingest documents from the docs directory"""
        if not self.docs_path.exists():
            logger.warning(f"Documents directory {self.docs_path} does not exist. Creating sample documents.")
            self._create_sample_documents()
        
        documents = []
        for file_path in self.docs_path.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'metadata': {'file_type': 'txt'}
                    })
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        # Process and store documents
        for doc in documents:
            self._process_and_store_document(doc)
        
        logger.info(f"Ingested {len(documents)} documents into vector database")
    
    def _create_sample_documents(self):
        """Create sample flood-related documents for testing"""
        self.docs_path.mkdir(exist_ok=True)
        
        sample_docs = {
            "flood_preparation_guide.txt": """
            FLOOD PREPARATION GUIDE FOR RESIDENTS
            
            Before a Flood:
            1. Create an emergency kit with:
               - Water (1 gallon per person per day for 3 days)
               - Non-perishable food for 3 days
               - Flashlight with extra batteries
               - First aid kit
               - Medications
               - Important documents in waterproof container
            
            2. Prepare your home:
               - Install check valves in plumbing
               - Waterproof basement walls
               - Clear gutters and downspouts
               - Move valuables to higher floors
            
            3. Know your evacuation route:
               - Identify multiple routes to higher ground
               - Practice evacuation with family
               - Keep vehicle fueled and ready
            
            During a Flood:
            1. Stay informed through local news and weather alerts
            2. Move to higher ground immediately if advised
            3. Do not walk through moving water
            4. Do not drive through flooded areas
            5. Turn off utilities if instructed
            
            After a Flood:
            1. Return home only when authorities say it's safe
            2. Avoid floodwaters - they may be contaminated
            3. Document damage for insurance claims
            4. Clean and disinfect everything that got wet
            """,
            
            "government_flood_response.txt": """
            GOVERNMENT FLOOD RESPONSE PROTOCOLS
            
            Emergency Response Levels:
            
            Level 1 - Monitoring:
            - Continuous monitoring of weather conditions
            - Regular updates to emergency services
            - Public information campaigns about preparedness
            
            Level 2 - Watch:
            - Flood conditions are possible
            - Emergency services on standby
            - Public advised to prepare emergency kits
            - Sandbag distribution centers activated
            
            Level 3 - Warning:
            - Flooding is imminent or occurring
            - Evacuation orders may be issued
            - Emergency shelters opened
            - Search and rescue teams deployed
            
            Level 4 - Emergency:
            - Major flooding in progress
            - Mass evacuation orders
            - State of emergency declared
            - Federal assistance requested
            
            Communication Protocols:
            - Regular press briefings
            - Social media updates every 2 hours
            - Emergency alert system activation
            - Coordination with neighboring jurisdictions
            
            Resource Management:
            - Sandbag distribution
            - Emergency shelter operations
            - Search and rescue coordination
            - Infrastructure damage assessment
            """,
            
            "response_team_procedures.txt": """
            EMERGENCY RESPONSE TEAM PROCEDURES
            
            First Responder Actions:
            
            Immediate Response (0-2 hours):
            1. Assess situation and report to command center
            2. Establish incident command post
            3. Begin search and rescue operations
            4. Set up communication networks
            5. Coordinate with other agencies
            
            Short-term Response (2-24 hours):
            1. Continue rescue operations
            2. Provide emergency medical care
            3. Establish temporary shelters
            4. Begin damage assessment
            5. Restore critical infrastructure
            
            Long-term Response (24+ hours):
            1. Transition to recovery operations
            2. Provide ongoing support to affected communities
            3. Conduct detailed damage assessments
            4. Coordinate volunteer efforts
            5. Plan for long-term recovery
            
            Safety Protocols:
            - Always work in teams of at least 2
            - Use proper personal protective equipment
            - Follow established communication protocols
            - Report all incidents immediately
            - Maintain detailed logs of all activities
            
            Equipment Requirements:
            - Water rescue equipment
            - Communication devices
            - Medical supplies
            - Safety equipment
            - Documentation tools
            """
        }
        
        for filename, content in sample_docs.items():
            file_path = self.docs_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _process_and_store_document(self, document: dict):
        """Process a document and store it in the vector database"""
        try:
            # Split document into chunks
            text_chunks = self.text_splitter.split_text(document['content'])
            
            # Create documents for each chunk
            docs = []
            for i, chunk in enumerate(text_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': document['source'],
                        'chunk_id': i,
                        **document.get('metadata', {})
                    }
                )
                docs.append(doc)
            
            # Generate embeddings and store
            for doc in docs:
                embedding = self.embedding_model.encode(doc.page_content).tolist()
                
                self.collection.add(
                    embeddings=[embedding],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata],
                    ids=[f"{doc.metadata['source']}_{doc.metadata['chunk_id']}"]
                )
                
        except Exception as e:
            logger.error(f"Error processing document {document['source']}: {e}")
    
    async def get_relevant_content(self, query: str, n_results: int = 5) -> Tuple[str, List[str]]:
        """
        Retrieve relevant content based on query using semantic search.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            Tuple of (combined_content, source_list)
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search for similar documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Combine results
            combined_content = ""
            sources = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    combined_content += f"\n\n{doc}"
                    if results['metadatas'] and results['metadatas'][0]:
                        source = results['metadatas'][0][i].get('source', 'Unknown')
                        if source not in sources:
                            sources.append(source)
            
            return combined_content.strip(), sources
            
        except Exception as e:
            logger.error(f"Error retrieving content for query '{query}': {e}")
            return "Unable to retrieve relevant information at this time.", []
    
    async def add_document(self, content: str, source: str, metadata: dict = None):
        """Add a new document to the vector database"""
        try:
            document = {
                'content': content,
                'source': source,
                'metadata': metadata or {}
            }
            self._process_and_store_document(document)
            logger.info(f"Successfully added document: {source}")
        except Exception as e:
            logger.error(f"Error adding document {source}: {e}")
            raise
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the document collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": "all-MiniLM-L6-v2"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
