import datetime
import logging
import os
from pydoc import doc
from typing import Optional
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint # Corrected import
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from reinforcement import ReinforcementLearner
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from operator import itemgetter
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
cohere_api_key = os.getenv("COHERE_API_KEY")

session_storage = defaultdict(list)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize HF Hub LLM - Corrected Initialization
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    task="text-generation",
    huggingfacehub_api_token=hf_api_token
)

output_parser = StrOutputParser()

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Provide clear and concise answers."),
    ("system", "Context: {context}"),
    ("system", "Chat History: {history}"),
    ("human", "{question}")
])

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Knowledge Graph Integration
try:
    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
    print("Successfully connected to Neo4j")
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    graph = None

# Initialize documents with a default document
default_docs = [Document(page_content="Initial document", metadata={"source": "default"})]

# Load or create FAISS vector store
try:
    vector_store = FAISS.load_local("prod_knowledge_base", embeddings, allow_dangerous_deserialization=True)
    print("Successfully loaded existing vector store")
except Exception as e:
    print(f"Creating new vector store: {e}")
    vector_store = FAISS.from_documents(default_docs, embeddings)
    vector_store.save_local("prod_knowledge_base")

# Define HyDE (Hypothetical Document Embeddings)
hyde_prompt = """Generate a hypothetical answer to: {question}"""

# Initialize the reinforcement learner
reinforcement_learner = ReinforcementLearner()

def format_docs(docs):
    """Formats retrieved documents into a single context string."""
    try:
        return "\n\n".join(
            str(doc.page_content) if hasattr(doc, 'page_content') else str(doc)
        )
    except Exception as e:
        logger.error(f"Error formatting documents: {e}")
        return ""

def format_chat_history(history):
    """Formats chat history into a string"""
    return "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])

def get_session_history(session_id):
    """Retrieves chat history for a session"""
    return session_storage.get(session_id, [])

def update_session_history(session_id, query, response):
    """Updates chat history for a session"""
    session_storage[session_id].append((query, response))

def format_chat_history(history: list) -> str:
    """Converts chat history to formatted string"""
    return "\n".join(
        f"User: {entry[0]}\nAssistant: {entry[1]}"
        for entry in history
    )

def hypothetical_doc_embeddings(query):
    hypothetical_answer = llm.invoke(hyde_prompt.format(question=query))
    return embeddings.embed_documents([hypothetical_answer])[0]

# Create advanced retriever
def create_advanced_retriever(vector_store):
    # Get raw documents for BM25
    raw_docs = vector_store.similarity_search("", k=100)
    bm25_retriever = BM25Retriever.from_documents(raw_docs)
    bm25_retriever.k = 5

    # Create vector store retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    try:
        compressor = CohereRerank(
            cohere_api_key=cohere_api_key,
            top_n=5,
            model='rerank-english-v2.0'
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        return compression_retriever
    except Exception as e:
        logging.warning(f"Falling back to ensemble retriever: {e}")
        return ensemble_retriever  # Remove any extra return statements

retriever = create_advanced_retriever(vector_store)

# Define Chat History Storage
class ChatAnalytics:
    def __init__(self):
        self.feedback_log = []

    def log_interaction(self, query, response, feedback=None):
        self.feedback_log.append({
            "timestamp": datetime.datetime.now(),
            "query": query,
            "response": response,
            "user_feedback": feedback
        })

    def calculate_accuracy(self):
        valid_feedback = [entry["user_feedback"] for entry in self.feedback_log
                         if isinstance(entry["user_feedback"], (int, float))]

        if not valid_feedback:
            return "No feedback available to calculate accuracy."

        average_score = sum(valid_feedback) / len(valid_feedback)
        max_score = 5  # Assuming feedback is rated from 1 to 5

        accuracy_percentage = (average_score / max_score) * 100
        return f"Chatbot Accuracy: {accuracy_percentage:.2f}% based on {len(valid_feedback)} feedback entries."

chat_analytics = ChatAnalytics()

### 1. Response Validation Decorator
def validate_response(func):
    """Ensures final output is always a valid string"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, dict):  # Handle dict outputs
                return json.dumps(result)
            return str(result)
        except Exception as e:
            logger.error(f"Response validation failed: {str(e)}")
            return "I need to reconsider how to answer that. Please try rephrasing your question."
    return wrapper

### 2. Updated Chain Builder with Type Safeguards
def build_production_chain(vector_store_or_retriever):
    # If you detect a vector store, convert it to a retriever
    if hasattr(vector_store_or_retriever, 'as_retriever'):
        retriever = create_advanced_retriever(vector_store_or_retriever)
    else:
        retriever = vector_store_or_retriever
    ...
    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | PROMPT_TEMPLATE
        | llm
        | output_parser
    ).with_config(run_name="ProductionQAChain")
    return chain

qa_chain = build_production_chain(retriever)

# Knowledge Graph QA Chain
kg_chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True) if graph else None

# Response Fusion
def response_fuser(base_response, kg_response=None):
    if kg_response:
        return f"Base Response: {base_response}\n\nKnowledge Graph Response: {kg_response}"
    return base_response

# Fallback Handler
def fallback_handler(query, confidence_score, timeouts):
    if confidence_score < 0.7:
        return "Could you rephrase that? I can help with: [Suggested Intents]"
    elif timeouts > 3:
        return "Let me connect you to a human specialist..."
    return "I'm not sure how to answer that, but I'm learning!"

# Continuous Learning Loop
def update_model(feedback_log):
    try:
        optimized_policy = reinforcement_learner.train(
            feedback_log,
            reward_fn=lambda x: x["user_feedback"]
        )
        if optimized_policy:
            #llm.update_weights(optimized_policy)
            pass
    except Exception as e:
        logging.error(f"Error updating model: {e}")

# Query Handling
def handle_query(query: str, history: list, qa_chain) -> str:
    """Process user queries with robust error handling"""
    try:
        if not query or not query.strip():
            return "Please provide a valid question."

        logger.info(f"Processing query: {query}")

        # Format the history properly
        formatted_history = format_chat_history(history) if history else ""

        # Invoke the chain with proper input structure
        response = qa_chain.invoke({
            "question": query.strip(),
            "history": formatted_history,
            "context": "" # context is handled in chain
        })

        # Ensure we have a string response
        if not isinstance(response, str):
            response = str(response)

        # Clean up any system artifacts
        response = response.replace("<|im_end|>", "").strip()

        logger.info(f"Generated response: {response[:100]}...")
        return response

    except Exception as e:
        logger.error(f"Error handling query: {str(e)}", exc_info=True)
        return "I apologize, but I'm having trouble processing your request. Please try again."

def chatbot():
    """Main chatbot function with proper initialization"""
    try:
        from chatbot_backend import build_production_chain, retriever

        # Initialize the QA chain
        qa_chain = build_production_chain(retriever)
        session_history = []

        print("Chatbot CLI - Type 'exit' to quit")
        while True:
            query = input("\nYou: ").strip()
            if query.lower() == 'exit':
                break

            response = handle_query(
                query=query,
                history=session_history,
                qa_chain=qa_chain
            )

            print(f"\nBot: {response}")
            session_history.append((query, response))

    except Exception as e:
        logger.error(f"Error in chatbot main loop: {str(e)}", exc_info=True)
        print("An error occurred. Please restart the chatbot.")

if __name__ == "__main__":
    chatbot()
