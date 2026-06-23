import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environments
load_dotenv()

# Configure logger
logger = logging.getLogger("FaceClustering.QdrantClient")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class QdrantFaceClient:
    """
    A client wrapper for Qdrant database to retrieve face embeddings and metadata.
    """
    def __init__(self):
        self.host = os.getenv("QDRANT_HOST", "192.168.1.125")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "faces")
        self.client: Optional[QdrantClient] = None
        
        # Initialize the connection
        self.connect()

    def connect(self) -> None:
        """
        Establish connection to the Qdrant server.
        """
        try:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}...")
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info("Successfully connected to Qdrant client.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}", exc_info=True)
            raise

    def fetch_all_face_vectors(self) -> List[Dict[str, Any]]:
        """
        Fetch all points (including vectors and payloads) from the configured collection.
        Uses scrolling to paginate through all points.
        
        Returns:
            List of dictionaries, each containing:
            - 'id': point ID (UUID or int)
            - 'vector': list of floats (embedding)
            - 'payload': dict containing metadata (like minio_url)
        """
        if not self.client:
            raise ConnectionError("Qdrant client is not connected.")
            
        logger.info(f"Fetching all face vectors from collection '{self.collection_name}'...")
        all_points = []
        next_page_offset = None
        
        try:
            # Check if collection exists first
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if self.collection_name not in collection_names:
                raise ValueError(f"Collection '{self.collection_name}' does not exist in Qdrant database.")
                
            while True:
                # Scroll points page by page
                response, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=True,
                    offset=next_page_offset
                )
                
                for point in response:
                    # Check if vector exists and payload exists
                    if point.vector is not None:
                        # Qdrant vector can be a list or a dictionary of named vectors.
                        # We extract the list of floats.
                        vector_data = point.vector
                        if isinstance(vector_data, dict):
                            # In case of multiple named vectors, we log warning and take the first one
                            # or check if there is a 'default' one.
                            logger.warning("Found named vectors in Qdrant. Using the first available vector.")
                            vector_data = next(iter(vector_data.values()))
                        
                        all_points.append({
                            "id": point.id,
                            "vector": vector_data,
                            "payload": point.payload or {}
                        })
                
                # If offset is None, we have reached the end of the collection
                if next_page_offset is None:
                    break
                    
            logger.info(f"Successfully fetched {len(all_points)} face vectors from Qdrant.")
            return all_points
            
        except Exception as e:
            logger.error(f"Error fetching points from Qdrant: {e}", exc_info=True)
            raise
