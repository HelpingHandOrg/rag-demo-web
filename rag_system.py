from semantic_search import SemanticSearchEngine
from openai import OpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with semantic search and LLM components."""
        self.search_engine = SemanticSearchEngine()
        
        # Get API key from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
    def _prepare_context(self, results: List[Dict]) -> str:
        """Prepare context from search results for the LLM."""
        context = "Here is information about relevant organizations:\n\n"
        
        for result in results:
            org_info = result['info']['organization_info']
            learning_paths = result['info']['learning_paths']
            
            context += f"""
Organization: {result['organization']}
- Platform: {org_info['Platform']}
- Status: {org_info['Subscription Status']}
- Service Level: {org_info['Service']}
- CNE Status: {org_info['CNE Status']}
- Billable Users: {org_info['Billable Users']}
- Organization Age: {org_info['Organization Age']} years
- Learning Paths: {learning_paths['Total Paths']} total
  * Completed: {learning_paths['Status Summary']['completed']}
  * Started: {learning_paths['Status Summary']['started']}
  * Joined: {learning_paths['Status Summary']['joined']}
- Available Services: {', '.join(learning_paths['Services'])}

"""
        return context

    def _generate_prompt(self, query: str, context: str) -> str:
        """Generate a prompt for the LLM using the query and context."""
        return f"""Based on the following information about organizations, please answer this question: "{query}"

{context}

Please provide a detailed answer using only the information provided above. If you can't answer the question completely with the given information, please state that explicitly.

Answer:"""

    def query(self, user_question: str, top_k: int = 3) -> Dict:
        """
        Process a user question using RAG:
        1. Retrieve relevant documents using semantic search
        2. Generate an answer using the LLM with retrieved context
        """
        # Retrieve relevant documents
        search_results = self.search_engine.search(user_question, top_k=top_k)
        
        # Prepare context from search results
        context = self._prepare_context(search_results)
        
        # Create a prompt combining the question and context
        prompt = self._generate_prompt(user_question, context)
        
        # Define system message
        system_message = "You are a helpful assistant that answers questions about organizations and their learning paths based on provided information. Only use the information provided in the context to answer questions."
        
        # Print the messages being sent to OpenAI
        print("\nSending to OpenAI:")
        print("-" * 80)
        print("System Message:")
        print(system_message)
        print("\nUser Prompt:")
        print(prompt)
        print("-" * 80)
        
        # Get answer from LLM using new API syntax
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            'answer': response.choices[0].message.content,
            'context': context,
            'sources': [
                {
                    'organization': result['organization'],
                    'similarity': result['similarity'],
                    'description': result['info']['description']
                }
                for result in search_results
            ]
        }

def main():
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Example questions
        questions = [
            "Which organizations have the highest learning path completion rates?",
            "What are the characteristics of organizations using the Enterprise platform?",
            "Compare the performance of organizations with Basic vs Enterprise subscriptions.",
            "Which organizations might need intervention due to low engagement?",
            "What patterns do you see in CNE usage across different platforms?"
        ]
        
        print("RAG System Examples:\n")
        
        for question in questions:
            print(f"\nQuestion: '{question}'")
            print("=" * 80)
            
            result = rag.query(question)
            
            print("\nAnswer:")
            print(result['answer'])
            
            print("\nSources:")
            for source in result['sources']:
                print(f"- {source['organization']} (Similarity: {source['similarity']:.3f})")
            
            print("\n" + "=" * 80 + "\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure your .env file is properly configured with OPENAI_API_KEY")

if __name__ == "__main__":
    main() 