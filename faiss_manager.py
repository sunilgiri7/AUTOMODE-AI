import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self, vector_store_path: str = "prod_knowledge_base"):
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self) -> FAISS:
        try:
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded existing vector store")
            return vector_store
        except Exception as e:
            logger.warning(f"Creating new vector store: {e}")
            return FAISS.from_texts(["Initialize vector store"], self.embeddings)

    def add_documents(self, file_paths: List[str], file_contents: Optional[List[str]] = None) -> Dict[str, Any]:
        documents = []
        failed_files = []
        
        # Process file paths
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        doc = Document(
                            page_content=content,
                            metadata={"source": str(file_path)}
                        )
                        documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                failed_files.append(str(file_path))
        
        # Process direct content strings
        if file_contents:
            for i, content in enumerate(file_contents):
                try:
                    doc = Document(
                        page_content=content,
                        metadata={"source": f"content_{i}"}
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error processing content chunk {i}: {e}")

        # Split and add documents
        if documents:
            try:
                splits = self.text_splitter.split_documents(documents)
                self.vector_store.add_documents(splits)
                self._save_vectorstore()
                logger.info(f"Added {len(splits)} document chunks to vector store")
                return {
                    "status": "success",
                    "chunks_added": len(splits),
                    "failed_files": failed_files
                }
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "failed_files": failed_files
                }
        
        return {
            "status": "error",
            "error": "No valid documents to process",
            "failed_files": failed_files
        }

    def _save_vectorstore(self) -> None:
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def get_vectorstore(self) -> FAISS:
        return self.vector_store

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []