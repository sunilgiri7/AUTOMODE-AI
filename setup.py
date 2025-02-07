import os
import logging

def setup_environment():
    """Create necessary directories and initialize the environment."""
    directories = [
        'prod_knowledge_base',
        'uploads',
        'logs'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created or verified directory: {directory}")
        except Exception as e:
            logging.error(f"Error creating directory {directory}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_environment()