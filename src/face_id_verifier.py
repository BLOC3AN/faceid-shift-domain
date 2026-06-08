import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from insightface.app import FaceAnalysis

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
        calibration_alpha: float = 0.6
    ):
        self.ref_image_path = ref_image_path
        self.model_root = model_root
        self.verify_threshold = verify_threshold
        self.calibration_enabled = calibration_enabled
        self.calibration_alpha = calibration_alpha
        
        self.app: Optional[FaceAnalysis] = None
        self.ref_embedding: Optional[np.ndarray] = None
        
        self._init_model()
        self._extract_ref_embedding()

    def _init_model(self) -> None:
        """Initialize local InsightFace models for the reference image only."""
        try:
            logger.info("Initializing FaceAnalysis for reference image extraction...")
            self.app = FaceAnalysis(name='buffalo_m', root=self.model_root)
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
        Runs verification directly on the provided cluster embeddings in RAM.
        
        Args:
            clusters: Dict mapping label (int) -> list of item dicts containing:
                      - "point_id"
                      - "vector"
        
        Returns:
            Dict containing:
                - "target_folder": Name of the folder representing the target cluster (e.g. 'cluster_1')
                - "delta_v_norm": Norm of the calculated translation vector
                - "matches_count": Total count of verified matches
                - "results": Dict mapping point_id -> {"similarity": float, "is_match": bool}
        """
        logger.info("Running FaceID verification directly on Qdrant vectors in RAM...")
        
        # Calculate raw similarities for all clusters
        raw_similarities = {}
        for label, items in clusters.items():
            folder_name = "noise" if label == -1 else f"cluster_{label}"
            raw_similarities[folder_name] = []
            for item in items:
                emb = np.array(item["vector"])
                norm_emb = emb / np.linalg.norm(emb)
                sim = self.cosine_similarity(self.ref_embedding, norm_emb)
                raw_similarities[folder_name].append(sim)
                
        # Identify target cluster (highest raw average similarity, excluding noise)
        target_folder = None
        target_label = None
        highest_avg_sim = -1.0
        
        for folder_name, sims in raw_similarities.items():
            if folder_name == "noise" or not sims:
                continue
            avg_sim = np.mean(sims)
            logger.info(f"Folder '{folder_name}' raw average similarity: {avg_sim:.4f}")
            if avg_sim > highest_avg_sim:
                highest_avg_sim = avg_sim
                target_folder = folder_name
                # Extract label from folder name (e.g. 'cluster_1' -> 1)
                target_label = int(folder_name.split("_")[1])
                
        # Calculate translation vector delta_v
        delta_v = None
        if self.calibration_enabled and target_label is not None:
            logger.info(f"Target cluster identified: '{target_folder}' with average similarity {highest_avg_sim:.4f}")
            target_embs = [np.array(x["vector"]) for x in clusters[target_label]]
            target_embs_norm = [x / np.linalg.norm(x) for x in target_embs]
            centroid = np.mean(target_embs_norm, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            delta_v = self.ref_embedding - centroid
            logger.info(f"Domain translation vector delta_v norm: {np.linalg.norm(delta_v):.4f}")
        else:
            logger.info("Domain calibration is disabled or no valid target cluster was found.")

        # Evaluate final calibrated similarities
        results_map = {}
        matches_count = 0
        
        for label, items in clusters.items():
            for item in items:
                point_id = item["point_id"]
                emb = np.array(item["vector"])
                norm_emb = emb / np.linalg.norm(emb)
                
                # Apply calibration if available
                if delta_v is not None:
                    calibrated_emb = norm_emb + self.calibration_alpha * delta_v
                    calibrated_emb = calibrated_emb / np.linalg.norm(calibrated_emb)
                    sim = self.cosine_similarity(self.ref_embedding, calibrated_emb)
                else:
                    sim = self.cosine_similarity(self.ref_embedding, norm_emb)
                    
                is_match = sim >= self.verify_threshold
                if is_match:
                    matches_count += 1
                    
                results_map[point_id] = {
                    "similarity": sim,
                    "is_match": is_match
                }
                
        logger.info(f"Verification mathematical analysis completed. Total matches found: {matches_count}.")
        
        return {
            "target_folder": target_folder,
            "delta_v_norm": float(np.linalg.norm(delta_v)) if delta_v is not None else 0.0,
            "results": results_map,
            "matches_count": matches_count
        }
