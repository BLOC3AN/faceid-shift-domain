import os
import cv2
import hashlib
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from insightface.app import FaceAnalysis
from redis_client import RedisCacheClient

logger = logging.getLogger("FaceClustering.FaceIDVerifier")

class FaceIDVerifier:
    """
    Handles pure mathematical FaceID matching and domain shift calibration
    directly on pre-extracted vectors, separating vector math from disk I/O.
    """
    def __init__(
        self, 
        ref_image_path: str, 
        model_root: str = ".", 
        verify_threshold: float = 0.65,
        calibration_enabled: bool = True,
        calibration_alpha: float = 0.6,
        redis_client: Optional[RedisCacheClient] = None
    ):
        self.ref_image_path = ref_image_path
        self.model_root = model_root
        self.verify_threshold = verify_threshold
        self.calibration_enabled = calibration_enabled
        self.calibration_alpha = calibration_alpha
        
        self.app: Optional[FaceAnalysis] = None
        self.ref_embedding: Optional[np.ndarray] = None
        self.redis_client = redis_client
        
        # Build cache key from ref image path
        path_hash = hashlib.md5(self.ref_image_path.encode()).hexdigest()[:12]
        self.cache_key = f"faceid:ref_embedding:{path_hash}"
        
        # Try loading cached embedding first (skip model init if hit)
        cached = self._load_from_cache()
        if cached is not None:
            self.ref_embedding = cached
            logger.info("✅ Reference embedding loaded from Redis cache (model init skipped).")
        else:
            logger.info("Cache miss — initializing model and extracting embedding...")
            self._init_model()
            self._extract_ref_embedding()
            self._save_to_cache()

    def _load_from_cache(self) -> Optional[np.ndarray]:
        """Try loading reference embedding from Redis cache."""
        if not self.redis_client or not self.redis_client.is_connected:
            return None
        return self.redis_client.load_embedding(self.cache_key)

    def _save_to_cache(self) -> None:
        """Save reference embedding to Redis cache."""
        if not self.redis_client or not self.redis_client.is_connected or self.ref_embedding is None:
            return
        self.redis_client.save_embedding(self.cache_key, self.ref_embedding)

    def clear_cache(self) -> None:
        """Delete cached embedding from Redis (call after successful pipeline)."""
        if not self.redis_client or not self.redis_client.is_connected:
            return
        if self.redis_client.delete(self.cache_key):
            logger.info(f"Redis cache cleared: {self.cache_key}")
        else:
            logger.debug(f"No cache entry to clear: {self.cache_key}")

    def _init_model(self) -> None:
        """Initialize local InsightFace models for the reference image only."""
        try:
            logger.info("Initializing FaceAnalysis (only detection and recognition) for reference image extraction...")
            self.app = FaceAnalysis(name='buffalo_m', root=self.model_root, allowed_modules=['detection', 'recognition'])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("FaceAnalysis initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize local FaceAnalysis model: {e}", exc_info=True)
            raise

    def _extract_ref_embedding(self) -> None:
        """Extract and normalize embedding for the reference image."""
        try:
            logger.info(f"Extracting embedding for reference image: {self.ref_image_path}")
            img = cv2.imread(self.ref_image_path)
            if img is None:
                raise FileNotFoundError(f"Reference image not found: {self.ref_image_path}")
                
            # Apply padding to assist reference face detection
            padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            faces = self.app.get(padded)
            if not faces:
                raise ValueError("No face detected in reference image!")
                
            # Pick largest face
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            emb = faces[0].embedding
            self.ref_embedding = emb / np.linalg.norm(emb)
            logger.info("Reference face embedding extracted and normalized successfully.")
        except Exception as e:
            logger.error(f"Failed to extract reference embedding: {e}", exc_info=True)
            raise

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two normalized vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def verify_vectors(self, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Runs vectorized verification on cluster embeddings in RAM using batch numpy ops.
        
        Args:
            clusters: Dict mapping label (int) -> list of item dicts containing:
                      - "point_id"
                      - "vector"
        
        Returns:
            Dict containing:
                - "target_folder": Name of the folder representing the target cluster
                - "delta_v_norm": Norm of the calculated translation vector
                - "matches_count": Total count of verified matches
                - "results": Dict mapping point_id -> {"similarity": float, "is_match": bool}
        """
        logger.info("Running vectorized FaceID verification on Qdrant vectors in RAM...")
        
        # Collect all vectors, point_ids, and labels into arrays
        all_ids = []
        all_labels = []
        all_vecs = []
        for label, items in clusters.items():
            for item in items:
                all_ids.append(item["point_id"])
                all_labels.append(label)
                all_vecs.append(item["vector"])
        
        if not all_vecs:
            return {"target_folder": None, "delta_v_norm": 0.0, "results": {}, "matches_count": 0}
        
        # Batch L2-normalize all vectors at once (single numpy operation)
        X = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X_norm = X / norms  # (N, D)
        labels_arr = np.array(all_labels)
        
        # Batch raw cosine similarities via matrix-vector multiplication
        raw_sims = X_norm @ self.ref_embedding  # (N,)
        
        # Identify target cluster (highest raw avg similarity, excluding noise)
        target_folder = None
        target_label = None
        highest_avg_sim = -1.0
        
        for label in set(all_labels):
            if label == -1:
                continue
            mask = labels_arr == label
            avg_sim = float(np.mean(raw_sims[mask]))
            folder_name = f"cluster_{label}"
            logger.info(f"Folder '{folder_name}' raw average similarity: {avg_sim:.4f}")
            if avg_sim > highest_avg_sim:
                highest_avg_sim = avg_sim
                target_folder = folder_name
                target_label = label
        
        # Calculate domain translation vector delta_v
        delta_v = None
        if self.calibration_enabled and target_label is not None:
            logger.info(f"Target cluster identified: '{target_folder}' with average similarity {highest_avg_sim:.4f}")
            target_mask = labels_arr == target_label
            centroid = np.mean(X_norm[target_mask], axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            delta_v = self.ref_embedding - centroid
            logger.info(f"Domain translation vector delta_v norm: {np.linalg.norm(delta_v):.4f}")
        else:
            logger.info("Domain calibration is disabled or no valid target cluster was found.")
        
        # Batch calibrated similarities via matrix-vector multiplication
        if delta_v is not None:
            X_calibrated = X_norm + self.calibration_alpha * delta_v
            cal_norms = np.linalg.norm(X_calibrated, axis=1, keepdims=True)
            X_calibrated = X_calibrated / cal_norms
            final_sims = X_calibrated @ self.ref_embedding  # (N,)
        else:
            final_sims = raw_sims
        
        # Batch threshold check
        is_match_arr = final_sims >= self.verify_threshold
        matches_count = int(np.sum(is_match_arr))
        
        # Build results map
        results_map = {}
        for i, point_id in enumerate(all_ids):
            results_map[point_id] = {
                "similarity": float(final_sims[i]),
                "is_match": bool(is_match_arr[i])
            }
        
        logger.info(f"Verification mathematical analysis completed. Total matches found: {matches_count}.")
        
        return {
            "target_folder": target_folder,
            "delta_v_norm": float(np.linalg.norm(delta_v)) if delta_v is not None else 0.0,
            "results": results_map,
            "matches_count": matches_count
        }
