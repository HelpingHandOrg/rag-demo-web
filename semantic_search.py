from sentence_transformers import SentenceTransformer
import numpy as np
from pymongo import MongoClient
import torch
from typing import List, Dict
import json
import streamlit as st
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def connect_to_mongodb():
    """Connect to MongoDB and return database instance."""
    try:
        mongodb_uri = st.secrets["MONGODB_URI"]
        if not mongodb_uri:
            raise ValueError("MONGODB_URI not found in secrets")
            
        print(f"Connecting to MongoDB...")
        client = MongoClient(mongodb_uri)
        # Test the connection
        client.server_info()
        print("Successfully connected to MongoDB!")
        
        db = client['rag_learning_db']
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        print(traceback.format_exc())
        raise

class SemanticSearchEngine:
    def __init__(self):
        """Initialize the semantic search engine with the model and database connection."""
        try:
            print("Initializing SentenceTransformer model...")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            if torch.cuda.is_available():
                print("CUDA is available, moving model to GPU...")
                self.model = self.model.to('cuda')
            print("Model initialized successfully!")
            
            print("Connecting to MongoDB...")
            self.db = connect_to_mongodb()
            print("Database connection established!")
            
        except Exception as e:
            print(f"Error initializing SemanticSearchEngine: {str(e)}")
            print(traceback.format_exc())
            raise
        
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a given text."""
        try:
            return self.model.encode(text, convert_to_tensor=True).cpu().numpy()
        except Exception as e:
            print(f"Error computing embedding: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def _compute_similarity(self, query_embedding: np.ndarray, doc_embedding: List[float]) -> float:
        """Compute cosine similarity between query and document embeddings."""
        try:
            return np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.3) -> List[Dict]:
        """
        Perform semantic search over the organization documents.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of top matching documents with their similarity scores
        """
        try:
            print(f"Processing search query: {query}")
            
            # Get query embedding
            query_embedding = self._compute_embedding(query)
            
            # Get all documents and compute similarities
            embeddings_collection = self.db['organization_embeddings']
            documents = list(embeddings_collection.find({}))
            print(f"Found {len(documents)} documents in database")
            
            results = []
            for doc in documents:
                if 'embedding' not in doc:
                    print(f"Warning: Document {doc.get('Organization Name', 'Unknown')} has no embedding")
                    continue
                    
                similarity = self._compute_similarity(query_embedding, doc['embedding'])
                
                if similarity >= min_similarity:
                    results.append({
                        'organization': doc['Organization Name'],
                        'similarity': float(similarity),
                        'info': {
                            'organization_info': doc['Organization Info'],
                            'learning_paths': doc['Learning Paths'],
                            'description': doc['Text Description']
                        }
                    })
            
            # Sort by similarity and get top k results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            print(f"Found {len(results)} results above similarity threshold")
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            print(traceback.format_exc())
            raise

def format_search_result(result: Dict) -> str:
    """Format a search result for display."""
    org_info = result['info']['organization_info']
    learning_paths = result['info']['learning_paths']
    
    return f"""
Organization: {result['organization']} (Similarity: {result['similarity']:.3f})
Platform: {org_info['Platform']}
Status: {org_info['Subscription Status']}
Billable Users: {org_info['Billable Users']}
Learning Paths: {learning_paths['Total Paths']} total
  - Completed: {learning_paths['Status Summary']['completed']}
  - Started: {learning_paths['Status Summary']['started']}
  - Joined: {learning_paths['Status Summary']['joined']}
"""

def main():
    try:
        # Initialize search engine
        search_engine = SemanticSearchEngine()
        
        # Example queries to demonstrate semantic search
        example_queries = [
            "organizations with many completed learning paths",
            "enterprise platform companies with high user engagement",
            "organizations with expired subscriptions and low completion rates",
            "new organizations with basic platform subscription",
            "organizations with most billable users"
        ]
        
        print("Semantic Search Examples:\n")
        
        for query in example_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 80)
            
            results = search_engine.search(query, top_k=3)
            
            for result in results:
                print(format_search_result(result))
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print("Please make sure your .env file is properly configured with MONGODB_URI")

if __name__ == "__main__":
    main()

app = FastAPI()
search_engine = SemanticSearchEngine()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    min_similarity: float = 0.3

@app.post("/search")
async def semantic_search(request: SearchRequest) -> List[Dict]:
    try:
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 