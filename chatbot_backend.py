import datetime
import logging
import os
import re
from typing import Optional, List, Dict, Any, Tuple
import json
from collections import defaultdict
from operator import itemgetter
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import GraphCypherQAChain
from langchain.docstore.document import Document

# Local imports
from reinforcement import ReinforcementLearner
from neo4j_manager import initialize_knowledge_graph

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotBackend:
    def __init__(self):
        self.session_storage = defaultdict(list)
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all necessary components for the chatbot"""
        # Initialize LLM
        self.llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            task="text-generation",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize knowledge graph
        self.graph = initialize_knowledge_graph()
        if self.graph:
            self.kg_chain = GraphCypherQAChain.from_llm(
                self.llm, 
                graph=self.graph, 
                verbose=True
            )
        else:
            self.kg_chain = None
            
        # Initialize retriever
        self.retriever = self._create_advanced_retriever()
        
        # Initialize other components
        self.output_parser = StrOutputParser()
        self.reinforcement_learner = ReinforcementLearner()
        self.chat_analytics = ChatAnalytics()
        
        # Initialize prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a highly specialized and professional AI assistant. "
                "Your task is to provide clear, concise, and factually accurate answers. "
                "Always base your response on the provided context and chat history, and ensure that your answer is directly relevant to the question. "
                "If the context does not provide enough information, explicitly state your uncertainty rather than guessing. "
                "Structure your answer in a logical, easy-to-read format (using bullet points or numbered lists when needed) without extraneous details."
            )),
            ("system", (
                "Instructions: Analyze the provided context thoroughly and verify your response for accuracy and consistency. "
                "Deliver your answer in a succinct manner using formal and neutral language. "
                "If multi-step reasoning is required, briefly outline your thought process, then provide the final answer. "
                "Avoid speculation, and if you are unsure, state that you need more context."
            )),
            ("system", "Context: {context}"),
            ("system", "Chat History: {history}"),
            ("human", "{question}")
        ])

    def _initialize_vector_store(self) -> FAISS:
        """Initialize or load FAISS vector store"""
        default_docs = [Document(page_content="Initial document", metadata={"source": "default"})]
        try:
            vector_store = FAISS.load_local(
                "prod_knowledge_base", 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded existing vector store")
        except Exception as e:
            logger.info(f"Creating new vector store: {e}")
            vector_store = FAISS.from_documents(default_docs, self.embeddings)
            vector_store.save_local("prod_knowledge_base")
        return vector_store

    def _create_advanced_retriever(self) -> EnsembleRetriever:
        """Create advanced retriever with ensemble and reranking"""
        # Initialize BM25 retriever
        raw_docs = self.vector_store.similarity_search("", k=100)
        bm25_retriever = BM25Retriever.from_documents(raw_docs)
        bm25_retriever.k = 5

        # Initialize vector retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4]
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
            logger.warning(f"Falling back to ensemble retriever: {e}")
            return ensemble_retriever

    def build_production_chain(self):
        """Build the production chain with all components"""
        chain = (
            {
                "context": RunnableLambda(lambda x: x["question"]) | self.retriever | self._format_docs,
                "question": itemgetter("question"),
                "history": itemgetter("history")
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        ).with_config(run_name="ProductionQAChain")
        return chain

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Format retrieved documents into a single context string"""
        try:
            return "\n\n".join(
                str(doc.page_content) if hasattr(doc, 'page_content') else str(doc)
                for doc in docs
            )
        except Exception as e:
            logger.error(f"Error formatting documents: {e}")
            return ""

    @staticmethod
    def _format_chat_history(history: List[Tuple[str, str]]) -> str:
        """Format chat history into a string"""
        return "\n".join(
            f"User: {entry[0]}\nAssistant: {entry[1]}"
            for entry in history
        )

    def get_session_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Retrieve chat history for a session"""
        return self.session_storage.get(session_id, [])

    def update_session_history(self, session_id: str, query: str, response: str):
        """Update chat history for a session"""
        self.session_storage[session_id].append((query, response))

    def handle_query(self, query: str, history: List[Tuple[str, str]], qa_chain) -> str:
        """Process user queries with robust error handling"""
        try:
            if not query or not query.strip():
                return "Please provide a valid question."

            logger.info(f"Processing query: {query}")
            
            # Format history
            formatted_history = self._format_chat_history(history) if history else ""

            # Get base response
            base_response = qa_chain.invoke({
                "question": query.strip(),
                "history": formatted_history,
                "context": ""
            })

            # Get knowledge graph response if available
            kg_response = None
            if self.kg_chain:
                try:
                    kg_response = self.kg_chain.run(query.strip())
                except Exception as e:
                    logger.error(f"Error getting KG response: {e}")

            # Fuse responses
            final_response = self._response_fuser(base_response, kg_response)
            
            # Clean up response
            final_response = final_response.replace("<|im_end|>", "").strip()
            
            logger.info(f"Generated response: {final_response[:100]}...")
            return final_response

        except Exception as e:
            logger.error(f"Error handling query: {str(e)}", exc_info=True)
            return "I apologize, but I'm having trouble processing your request. Please try again."

    def _response_fuser(self, base_response: str, kg_response: Optional[str] = None) -> str:
        """Fuse responses from different sources"""
        if kg_response:
            return f"Base Response: {base_response}\n\nKnowledge Graph Response: {kg_response}"
        return base_response

    def update_model(self, feedback_log: List[Dict[str, Any]]):
        """Update model based on feedback"""
        try:
            optimized_policy = self.reinforcement_learner.train(
                feedback_log,
                reward_fn=lambda x: x["user_feedback"]
            )
            if optimized_policy:
                # Model update logic here
                pass
        except Exception as e:
            logger.error(f"Error updating model: {e}")

class EnhancedNeo4jManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.graph = self._initialize_graph()
        
    def _initialize_graph(self) -> Optional[Neo4jGraph]:
        try:
            # First, check and drop existing indexes/constraints
            self._cleanup_existing_schema()
            
            # Then create new schema
            self._setup_schema()
            
            graph = Neo4jGraph(
                url=self.uri,
                username=self.user,
                password=self.password,
                database=self.database
            )
            return graph
        except Exception as e:
            logger.error(f"Failed to initialize Neo4jGraph: {e}")
            return None
    
    def _cleanup_existing_schema(self):
        """Clean up existing indexes and constraints"""
        with self.driver.session(database=self.database) as session:
            try:
                # Drop existing indexes
                session.run("SHOW INDEXES YIELD name, labelsOrTypes, properties WHERE labelsOrTypes = ['Document'] OR labelsOrTypes = ['Entity'] WITH name CALL db.index.drop(name) YIELD state RETURN state")
                
                # Drop existing constraints
                session.run("SHOW CONSTRAINTS YIELD name WITH name CALL db.constraints.drop(name) YIELD name RETURN name")
                
                logger.info("Successfully cleaned up existing schema")
            except Exception as e:
                logger.error(f"Error cleaning up schema: {e}")
            
    def _setup_schema(self):
        """Set up necessary constraints and indexes"""
        with self.driver.session(database=self.database) as session:
            try:
                # Create constraints
                session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                
                # Create indexes
                session.run("CREATE INDEX document_content IF NOT EXISTS FOR (d:Document) ON (d.content)")
                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                
                logger.info("Successfully set up schema")
            except Exception as e:
                logger.error(f"Error setting up schema: {e}")

    def semantic_search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search using document and entity relationships"""
        cypher_query = """
        MATCH (doc:Document)
        WHERE doc.content CONTAINS $query_text
        MATCH (doc)-[:CONTAINS]->(e:Entity)
        WITH doc, e
        ORDER BY doc.id
        RETURN DISTINCT doc.title as title, 
               doc.content as content,
               collect(DISTINCT {name: e.name, type: e.type}) as entities
        LIMIT $limit
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                params = {
                    "query_text": query_text,
                    "limit": limit
                }
                result = session.run(cypher_query, params)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

class EnhancedResponseHandler:
    def __init__(self):
        self.response_history = []
        
    def clean_response(self, response: str) -> str:
        """Clean and format the response to ensure coherency"""
        if not response:
            return "I apologize, but I couldn't generate a proper response."
            
        # Remove repeated patterns
        cleaned = re.sub(r'(.{50,}?)\1+', r'\1', response)
        
        # Remove version numbers and technical IDs
        cleaned = re.sub(r'Python: \d+\.\d+\.\d+\.\d+', '', cleaned)
        cleaned = re.sub(r'\+\d+\s*\d*', '', cleaned)
        
        # Remove garbage patterns
        cleaned = re.sub(r'[A-Z]:\s*(?:[A-Za-z]+\s*)+(?:in a|to work|with our|professional)\s*', '', cleaned)
        
        # Clean up multiple spaces and newlines
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        return cleaned.strip()
    
    def format_response(self, clean_text: str) -> str:
        """Format the cleaned response into a proper structure"""
        if len(clean_text) < 10:
            return "Hello! How can I assist you today?"
            
        paragraphs = [p.strip() for p in clean_text.split('\n') if p.strip()]
        formatted = '\n\n'.join(paragraphs)
        
        return formatted

class EnhancedChatbotBackend(ChatbotBackend):
    def __init__(self):
        super().__init__()
        self.neo4j_manager = EnhancedNeo4jManager()
        self.response_handler = EnhancedResponseHandler()
        
    def handle_query(self, query: str, history: List[Tuple[str, str]], qa_chain) -> str:
        """Enhanced query handling with improved response cleaning"""
        try:
            # Get base response
            base_response = super().handle_query(query, history, qa_chain)
            
            # Clean and format the base response
            cleaned_response = self.response_handler.clean_response(base_response)
            formatted_response = self.response_handler.format_response(cleaned_response)
            
            # Only add knowledge graph info if we have a valid base response
            if len(formatted_response) > 20:
                try:
                    kg_results = self.neo4j_manager.semantic_search(query)
                    if kg_results:
                        kg_response = self._format_kg_response(kg_results)
                        formatted_response += f"\n\n{kg_response}"
                except Exception as e:
                    logger.error(f"Error getting knowledge graph response: {e}")
            
            # Final validation
            if not self._is_valid_response(formatted_response):
                return "Hi! I'm here to help. Could you please rephrase your question?"
                
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in enhanced query handling: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error. How else can I assist you?"
    
    def _format_kg_response(self, kg_results: List[Dict[str, Any]]) -> str:
        """Format knowledge graph results in a clean way"""
        if not kg_results:
            return ""
            
        response = "Additional Information:\n"
        for result in kg_results:
            if 'title' in result and 'entities' in result:
                response += f"• {result['title']}\n"
                for entity in result['entities']:
                    if isinstance(entity, dict) and 'name' in entity and 'type' in entity:
                        response += f"  - {entity['name']} ({entity['type']})\n"
                        
        return response.strip()
    
    def _is_valid_response(self, response: str) -> bool:
        """Validate if the response meets quality criteria"""
        if not response:
            return False
            
        if len(response) < 10:
            return False
            
        if re.search(r'(.{20,}?)\1', response):
            return False
            
        garbage_patterns = [
            r'\d{4}\s*\+\s*\d{4}',
            r'Python:\s*\d+\.\d+',
            r'[A-Z]:\s*(?:[A-Za-z]+\s*){3,}',
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, response):
                return False
                
        return True

class ResponseFormatter:
    @staticmethod
    def format_text(text: str) -> str:
        """Format plain text into well-structured paragraphs"""
        # Split into paragraphs and clean
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Format each paragraph
        formatted_paragraphs = []
        for p in paragraphs:
            # Check if it's a list
            if re.match(r'^[-*]\s', p):
                # Format as bullet points
                items = [line.strip() for line in p.split('\n') if line.strip()]
                formatted_paragraphs.append('\n'.join(f"• {item[2:]}" if item.startswith('- ') else f"• {item[2:]}" if item.startswith('* ') else f"• {item}" for item in items))
            # Check if it's code
            elif '```' in p:
                # Preserve code formatting
                formatted_paragraphs.append(p)
            else:
                # Regular paragraph
                formatted_paragraphs.append(p)
        
        return '\n\n'.join(formatted_paragraphs)

    @staticmethod
    def format_code(code: str) -> str:
        """Format code blocks with proper indentation"""
        if '```' not in code:
            return code
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', code, re.DOTALL)
        formatted_code = code
        
        for block in code_blocks:
            # Clean and indent code
            cleaned_block = block.strip()
            lines = cleaned_block.split('\n')
            # Determine base indentation
            base_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
            # Remove base indentation and reindent
            formatted_block = '\n'.join(' ' * 4 + line[base_indent:] for line in lines)
            # Replace in original text
            formatted_code = formatted_code.replace(block, formatted_block)
        
        return formatted_code
    
class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def create_knowledge_graph(self, documents: List[Dict[str, Any]]):
        """Create knowledge graph from documents"""
        with self.driver.session() as session:
            for doc in documents:
                # Create document node
                session.run("""
                    MERGE (d:Document {id: $id})
                    SET d.content = $content, d.title = $title
                """, id=doc['id'], content=doc['content'], title=doc['title'])
                
                # Create entity nodes and relationships
                for entity in doc['entities']:
                    session.run("""
                    MERGE (e:Entity {name: $name, type: $type})
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:CONTAINS]->(e)
                    """, name=entity['name'], type=entity['type'], doc_id=doc['id'])
                    
                # Create relationship nodes
                for rel in doc['relationships']:
                    session.run("""
                    MATCH (e1:Entity {name: $entity1})
                    MATCH (e2:Entity {name: $entity2})
                    MERGE (e1)-[:RELATES {type: $rel_type}]->(e2)
                    """, entity1=rel['entity1'], entity2=rel['entity2'], rel_type=rel['type'])

    def query_knowledge_graph(self, query: str, params: Optional[dict] = None):
        # Use params when running the query
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return result.data()  # or any other processing you need

class ChatAnalytics:
    def __init__(self):
        self.feedback_log = []

    def log_interaction(self, query: str, response: str, feedback: Optional[float] = None):
        """Log chat interaction with optional feedback"""
        self.feedback_log.append({
            "timestamp": datetime.datetime.now(),
            "query": query,
            "response": response,
            "user_feedback": feedback
        })

    def calculate_accuracy(self) -> str:
        """Calculate chatbot accuracy based on feedback"""
        valid_feedback = [
            entry["user_feedback"] for entry in self.feedback_log
            if isinstance(entry["user_feedback"], (int, float))
        ]

        if not valid_feedback:
            return "No feedback available to calculate accuracy."

        average_score = sum(valid_feedback) / len(valid_feedback)
        max_score = 5
        accuracy_percentage = (average_score / max_score) * 100
        
        return f"Chatbot Accuracy: {accuracy_percentage:.2f}% based on {len(valid_feedback)} feedback entries."

# Initialize global instance
# chatbot = ChatbotBackend()
chatbot = EnhancedChatbotBackend()
qa_chain = chatbot.build_production_chain()