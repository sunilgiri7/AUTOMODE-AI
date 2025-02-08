from langchain_neo4j import Neo4jGraph  # Updated import
from neo4j import GraphDatabase
import logging
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class Neo4jConnection:
    def __init__(self):
        self.driver = None
        self.graph = None

    def connect(self, 
            uri: str = os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username: str = os.getenv('NEO4J_USER', 'neo4j'),
            password: str = os.getenv('NEO4J_PASSWORD', 'password'),
            database: str = os.getenv('NEO4J_DATABASE', 'neo4j')):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test the connection
            with self.driver.session(database=database) as session:
                # Check if APOC is available using the updated command for Neo4j 5
                result = session.run("SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN count(*) as count")
                apoc_count = result.single()["count"]
                
                if apoc_count == 0:
                    logger.warning("APOC procedures not found. Some features may be limited.")
                else:
                    logger.info(f"Found {apoc_count} APOC procedures")
                
                # Remove this index creation to avoid conflict with the unique constraint:
                # session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                
            logger.info("Successfully connected to Neo4j database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

    def close(self):
        if self.driver:
            self.driver.close()

def initialize_knowledge_graph() -> Optional[Neo4jGraph]:
    neo4j_conn = Neo4jConnection()
    if neo4j_conn.connect():
        try:
            # Configure Neo4j with basic setup if APOC is not available
            with neo4j_conn.driver.session() as session:
                # Create basic schema constraints
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                
                # Create basic indexes for performance
                session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.content)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            
            # Initialize Neo4jGraph with fallback options
            graph = Neo4jGraph(
                url=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                username=os.getenv('NEO4J_USER', 'neo4j'),
                password=os.getenv('NEO4J_PASSWORD', 'password'),
                database=os.getenv('NEO4J_DATABASE', 'neo4j')
            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Error initializing Neo4j Graph: {e}")
            # Instead of returning the driver (which lacks get_structured_schema), return None
            return None
            
    return None