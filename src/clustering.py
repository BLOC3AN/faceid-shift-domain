import logging
import numpy as np
from typing import List, Tuple
from sklearn.cluster import HDBSCAN

# Configure logger
logger = logging.getLogger("FaceClustering.Clustering")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class FaceClustering:
    """
    Handles face embedding clustering using HDBSCAN algorithm.
    """
    def __init__(self, min_cluster_size: int = 3, min_samples: int = 1, metric: str = "euclidean"):
        """
        Initialize HDBSCAN clustering parameters.
        
        Args:
            min_cluster_size (int): The minimum size of a cluster. Default is 3.
            min_samples (int): The number of samples in a neighborhood for a point
                               to be considered as a core point. Default is 1 to allow smaller clusters.
            metric (str): The metric to use when calculating distance between instances.
                          Default is 'euclidean' (which works perfectly on L2-normalized embeddings).
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform L2 normalization on face embeddings to prepare them for Euclidean distance metrics.
        This is crucial because cosine distance corresponds to euclidean distance of L2-normalized vectors.
        
        Args:
            embeddings (np.ndarray): Shape (N, D) embeddings matrix.
            
        Returns:
            np.ndarray: Normalized embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Prevent division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def run(self, embeddings_list: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs HDBSCAN clustering on the provided list of embeddings.
        
        Args:
            embeddings_list (List[List[float]]): List of embeddings.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - labels: Cluster labels for each embedding. -1 represents noise/outliers.
                - probabilities: Cluster assignment probabilities/confidence scores.
        """
        if not embeddings_list:
            logger.warning("No embeddings provided for clustering.")
            return np.array([]), np.array([])
            
        logger.info(f"Preparing {len(embeddings_list)} embeddings for clustering...")
        X = np.array(embeddings_list, dtype=np.float32)
        
        # Normalize embeddings to ensure high quality distance metrics
        X_norm = self._normalize_embeddings(X)
        
        logger.info(
            f"Running HDBSCAN (min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}, metric='{self.metric}')..."
        )
        
        try:
            hdb = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                store_centers="centroid"
            )
            hdb.fit(X_norm)
            
            labels = hdb.labels_
            probabilities = hdb.probabilities_
            
            # Count clusters
            unique_labels = set(labels)
            num_clusters = len(unique_labels - {-1})
            num_noise = list(labels).count(-1)
            
            logger.info(f"Clustering complete. Found {num_clusters} clusters. Noise points: {num_noise}/{len(labels)}")
            return labels, probabilities
            
        except Exception as e:
            logger.error(f"Error occurred during HDBSCAN clustering: {e}", exc_info=True)
            raise
