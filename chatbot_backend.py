import datetime
import logging
import os
import re
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from operator import itemgetter

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.docstore.document import Document

# Local imports
from reinforcement import ReinforcementLearner

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextManager:
    """Manages context and knowledge base for the chatbot"""
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.recent_contexts = []
        self.max_context_length = 5
        self.vector_store = None
        
    def initialize_vector_store(self) -> FAISS:
        """Initialize or load FAISS vector store with better error handling"""
        try:
            self.vector_store = FAISS.load_local(
                "prod_knowledge_base",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded existing vector store")
        except Exception as e:
            logger.warning(f"Creating new vector store: {e}")
            default_docs = [
                Document(
                    page_content="Initial system knowledge",
                    metadata={"source": "default", "timestamp": datetime.datetime.now()}
                )
            ]
            self.vector_store = FAISS.from_documents(default_docs, self.embeddings)
            self.vector_store.save_local("prod_knowledge_base")
        return self.vector_store
    
    def update_context(self, new_context: str):
        """Update recent contexts with new information"""
        self.recent_contexts.append({
            "content": new_context,
            "timestamp": datetime.datetime.now()
        })
        if len(self.recent_contexts) > self.max_context_length:
            self.recent_contexts.pop(0)
    
    def get_relevant_context(self, query: str) -> str:
        """Get relevant context based on query"""
        if not self.vector_store:
            return ""
        
        # Combine recent contexts with vector store results
        relevant_docs = self.vector_store.similarity_search(query, k=3)
        recent_context = "\n".join(ctx["content"] for ctx in self.recent_contexts[-3:])
        
        return f"{recent_context}\n\n" + "\n".join(doc.page_content for doc in relevant_docs)

class SessionManager:
    """Manages chat sessions and history"""
    def __init__(self):
        self.sessions = defaultdict(list)
        self.session_metadata = defaultdict(dict)
        self.max_history_length = 10
        
    def add_interaction(self, session_id: str, query: str, response: str):
        """Add new interaction to session history"""
        self.sessions[session_id].append({
            "query": query,
            "response": response,
            "timestamp": datetime.datetime.now()
        })
        
        # Trim history if too long
        if len(self.sessions[session_id]) > self.max_history_length:
            self.sessions[session_id].pop(0)
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get formatted session history"""
        return self.sessions.get(session_id, [])
    
    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Update session metadata"""
        self.session_metadata[session_id].update(metadata)

class ImprovedResponseHandler:
    """Handles response generation and formatting with improved validation"""
    def __init__(self):
        self.response_patterns = {
            'greeting': r'^(hi|hello|hey)',
            'question': r'\?$',
            'command': r'^(show|tell|give|explain)',
        }
        
    def clean_response(self, response: str) -> str:
        """Clean response with improved handling"""
        if not response:
            return ""
            
        # Remove repeated segments more carefully
        cleaned = re.sub(r'(.{30,}?)\1+', r'\1', response)
        
        # Clean technical artifacts
        cleaned = re.sub(r'<\|.*?\|>', '', cleaned)
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def format_response(self, response: str, query: str) -> str:
        """Format response based on query type and content"""
        response = self.clean_response(response)
        
        if not response:
            return self._generate_fallback_response(query)
        
        # Identify query type
        query_type = self._identify_query_type(query.lower())
        
        # Format based on query type
        if query_type == 'greeting':
            return self._format_greeting(response)
        elif query_type == 'question':
            return self._format_answer(response)
        elif query_type == 'command':
            return self._format_instruction_response(response)
        
        return response
    
    def _identify_query_type(self, query: str) -> str:
        """Identify the type of query"""
        for pattern_type, pattern in self.response_patterns.items():
            if re.search(pattern, query):
                return pattern_type
        return 'general'
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate contextual fallback response"""
        if re.search(self.response_patterns['greeting'], query.lower()):
            return "Hello! How can I assist you today?"
        elif re.search(self.response_patterns['question'], query):
            return "I understand you have a question. Could you please provide more details?"
        return "I'm here to help. Could you please elaborate on what you need?"
    
    def _format_greeting(self, response: str) -> str:
        """Format greeting response"""
        return response.capitalize()
    
    def _format_answer(self, response: str) -> str:
        """Format answer response"""
        if not response.strip().endswith(('.', '!', '?')):
            response += '.'
        return response
    
    def _format_instruction_response(self, response: str) -> str:
        """Format instruction response"""
        paragraphs = response.split('\n\n')
        formatted = '\n\n'.join(p.strip() for p in paragraphs if p.strip())
        return formatted

class EnhancedChatbotBackend:
    """Main chatbot class with improved context and response management"""
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.context_manager = ContextManager(self.embeddings)
        self.session_manager = SessionManager()
        self.response_handler = ImprovedResponseHandler()
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize chatbot components"""
        # Initialize LLM
        self.llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            task="text-generation",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        # Initialize vector store and retriever
        self.vector_store = self.context_manager.initialize_vector_store()
        self.retriever = self._create_advanced_retriever()
        
        # Initialize prompt template with improved context handling
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
                You are a helpful AI assistant. Provide clear, relevant answers based on the context and chat history.
                If uncertain, acknowledge it and ask for clarification. Use natural, conversational language while maintaining professionalism.
                
                Guidelines:
                - Focus on addressing the specific query
                - Use relevant information from context and history
                - Be concise but thorough
                - Maintain a consistent tone
            """),
            ("system", "Context: {context}"),
            ("system", "Recent History: {history}"),
            ("human", "{question}")
        ])
    
    def _create_advanced_retriever(self) -> EnsembleRetriever:
        """Create advanced retriever with improved weights"""
        bm25_retriever = BM25Retriever.from_documents(
            self.vector_store.similarity_search("", k=100),
            k=5
        )
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # Adjusted weights for better relevance
        )
        
        try:
            compressor = CohereRerank(
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                top_n=5,
                model='rerank-english-v2.0'
            )
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
        except Exception as e:
            logger.warning(f"Using ensemble retriever without reranking: {e}")
            return ensemble_retriever
    
    def build_chain(self):
        """Build the processing chain with improved context handling"""
        return (
            {
                "context": RunnableLambda(self._get_context),
                "question": itemgetter("question"),
                "history": RunnableLambda(self._get_history)
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        ).with_config({"run_name": "ImprovedQAChain"})
    
    def _get_context(self, inputs: Dict[str, Any]) -> str:
        """Get relevant context for the query"""
        query = inputs["question"]
        session_id = inputs.get("session_id", "default")
        
        # Get context from both recent history and vector store
        relevant_context = self.context_manager.get_relevant_context(query)
        
        # Update context with current query
        self.context_manager.update_context(query)
        
        return relevant_context
    
    def _get_history(self, inputs: Dict[str, Any]) -> str:
        """Get formatted chat history"""
        session_id = inputs.get("session_id", "default")
        history = self.session_manager.get_session_history(session_id)
        
        if not history:
            return ""
        
        # Format recent history entries
        formatted_history = "\n".join(
            f"User: {entry['query']}\nAssistant: {entry['response']}"
            for entry in history[-3:]  # Only use last 3 interactions
        )
        
        return formatted_history
    
    def handle_query(self, query: str, session_id: str = "default") -> str:
        """Process query with improved context and response handling"""
        try:
            if not query.strip():
                return "Please provide a question or request."
            
            # Build and run chain
            chain = self.build_chain()
            base_response = chain.invoke({
                "question": query.strip(),
                "session_id": session_id
            })
            
            # Process and format response
            final_response = self.response_handler.format_response(base_response, query)
            
            # Update session history
            self.session_manager.add_interaction(session_id, query, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return "I encountered an issue processing your request. Could you please try again?"

# Initialize global instance
chatbot = EnhancedChatbotBackend()