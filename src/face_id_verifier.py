import os
import shutil
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from insightface.app import FaceAnalysis

logger = logging.getLogger("FaceClustering.FaceIDVerifier")

class FaceIDVerifier:
    """
    Handles FaceID matching and domain shift calibration directly using
    pre-extracted vectors fetched from Qdrant, avoiding expensive local re-embedding.
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

    def verify_vectors(self, clustered_items: Dict[str, List[Dict[str, Any]]], base_data_dir: str) -> Dict[str, Any]:
        """
        Runs verification directly on the provided Qdrant vectors.
        
        Args:
            clustered_items: Dict mapping folder_name -> list of item dicts containing:
                             - "point_id"
                             - "vector"
                             - "local_path"
                             - "minio_url"
            base_data_dir: Output base directory.
        """
        logger.info("Running FaceID verification directly on Qdrant vectors...")
        
        # Calculate raw similarities for all clusters
        raw_similarities = {}
        for folder_name, items in clustered_items.items():
            raw_similarities[folder_name] = []
            for item in items:
                emb = np.array(item["vector"])
                norm_emb = emb / np.linalg.norm(emb)
                sim = self.cosine_similarity(self.ref_embedding, norm_emb)
                raw_similarities[folder_name].append(sim)
                
        # Identify target cluster (highest raw average similarity, excluding noise)
        target_folder = None
        highest_avg_sim = -1.0
        
        for folder_name, sims in raw_similarities.items():
            if folder_name == "noise" or not sims:
                continue
            avg_sim = np.mean(sims)
            logger.info(f"Folder '{folder_name}' raw average similarity: {avg_sim:.4f}")
            if avg_sim > highest_avg_sim:
                highest_avg_sim = avg_sim
                target_folder = folder_name
                
        # Calculate translation vector delta_v
        delta_v = None
        if self.calibration_enabled and target_folder:
            logger.info(f"Target cluster identified: '{target_folder}' with average similarity {highest_avg_sim:.4f}")
            target_embs = [np.array(x["vector"]) for x in clustered_items[target_folder]]
            target_embs_norm = [x / np.linalg.norm(x) for x in target_embs]
            centroid = np.mean(target_embs_norm, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            delta_v = self.ref_embedding - centroid
            logger.info(f"Domain translation vector delta_v norm: {np.linalg.norm(delta_v):.4f}")
        else:
            logger.info("Domain calibration is disabled or no valid target cluster was found.")

        # Evaluate final calibrated similarities
        verified_results = {}
        matches_count = 0
        
        # Prepare match destination directory
        ref_basename = os.path.splitext(os.path.basename(self.ref_image_path))[0]
        match_dest_dir = os.path.join(base_data_dir, "matches", ref_basename)
        if os.path.exists(match_dest_dir):
            shutil.rmtree(match_dest_dir)
        os.makedirs(match_dest_dir, exist_ok=True)

        for folder_name, items in clustered_items.items():
            verified_results[folder_name] = []
            for item in items:
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
                item_result = {
                    "filename": os.path.basename(item["local_path"]),
                    "path": item["local_path"],
                    "similarity": sim,
                    "is_match": is_match
                }
                verified_results[folder_name].append(item_result)
                
                if is_match and os.path.exists(item["local_path"]):
                    matches_count += 1
                    # Copy to matches folder
                    shutil.copy2(item["local_path"], os.path.join(match_dest_dir, os.path.basename(item["local_path"])))
                    
            # Sort results by similarity descending
            verified_results[folder_name] = sorted(verified_results[folder_name], key=lambda x: x["similarity"], reverse=True)
            
        logger.info(f"Verification completed. Total matches found: {matches_count}. Matched images copied to '{match_dest_dir}'.")
        
        return {
            "target_folder": target_folder,
            "delta_v_norm": float(np.linalg.norm(delta_v)) if delta_v is not None else 0.0,
            "results": verified_results,
            "matches_count": matches_count,
            "match_dest_dir": match_dest_dir
        }
