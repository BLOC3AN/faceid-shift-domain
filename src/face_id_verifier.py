import os
import glob
import shutil
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from insightface.app import FaceAnalysis

logger = logging.getLogger("FaceClustering.FaceIDVerifier")

class FaceIDVerifier:
    """
    Handles local alignment, re-embedding, and domain shift calibration
    to match clustered images against a reference face.
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
        """Initialize local InsightFace models."""
        try:
            logger.info("Initializing FaceAnalysis for local verification...")
            self.app = FaceAnalysis(name='buffalo_m', root=self.model_root)
            # Use CPU execution provider (-1)
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
                
            # Apply padding just in case the reference is cropped tight
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

    def extract_image_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Align and extract normalized face embedding from an image."""
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None
                
            # Pad image by 50px to assist SCRFD detection near boundaries
            padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            faces = self.app.get(padded)
            if not faces:
                return None
                
            # Pick largest face
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            emb = faces[0].embedding
            return emb / np.linalg.norm(emb)
        except Exception as e:
            logger.error(f"Error extracting embedding from {file_path}: {e}")
            return None

    def run_verification(self, base_data_dir: str) -> Dict[str, Any]:
        """
        Runs local verification on all clusters:
        1. Evaluates raw similarities.
        2. Automatically identifies the target cluster (highest raw average similarity).
        3. Calculates domain translation vector.
        4. Applies calibration and filters matches.
        """
        logger.info("Starting local FaceID verification and domain calibration...")
        
        # Scan cluster directories
        cluster_dirs = sorted([d for d in glob.glob(os.path.join(base_data_dir, "cluster_*")) if os.path.isdir(d)])
        noise_dir = os.path.join(base_data_dir, "noise")
        if os.path.exists(noise_dir) and os.path.isdir(noise_dir):
            cluster_dirs.append(noise_dir)
            
        all_embeddings = {}  # folder_name -> list of (filename, path, emb)
        raw_similarities = {}  # folder_name -> list of similarities
        
        # Extract embeddings for all images
        for folder_path in cluster_dirs:
            folder_name = os.path.basename(folder_path)
            images = glob.glob(os.path.join(folder_path, "*.jpg"))
            
            all_embeddings[folder_name] = []
            raw_similarities[folder_name] = []
            
            logger.info(f"Extracting embeddings for folder '{folder_name}' ({len(images)} images)...")
            for img_path in images:
                emb = self.extract_image_embedding(img_path)
                if emb is not None:
                    sim = self.cosine_similarity(self.ref_embedding, emb)
                    all_embeddings[folder_name].append({
                        "filename": os.path.basename(img_path),
                        "path": img_path,
                        "embedding": emb
                    })
                    raw_similarities[folder_name].append(sim)
                    
        # Identify target cluster based on highest average similarity (excluding noise)
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
                
        # Calculate translation vector if calibration is enabled and a target cluster is found
        delta_v = None
        if self.calibration_enabled and target_folder:
            logger.info(f"Target cluster identified: '{target_folder}' with average similarity {highest_avg_sim:.4f}")
            target_embs = [x["embedding"] for x in all_embeddings[target_folder]]
            if target_embs:
                centroid = np.mean(target_embs, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                delta_v = self.ref_embedding - centroid
                logger.info(f"Domain translation vector calculated. Norm: {np.linalg.norm(delta_v):.4f}")
        else:
            logger.info("Domain calibration is disabled or no valid target cluster was found.")

        # Evaluate final calibrated similarities
        verified_results = {}
        matches_count = 0
        
        # Prepare match destination directory
        ref_basename = os.path.splitext(os.path.basename(self.ref_image_path))[0]
        # E.g. data/matches/duong_cropped/
        match_dest_dir = os.path.join(base_data_dir, "matches", ref_basename)
        if os.path.exists(match_dest_dir):
            shutil.rmtree(match_dest_dir)
        os.makedirs(match_dest_dir, exist_ok=True)

        for folder_name, items in all_embeddings.items():
            verified_results[folder_name] = []
            for item in items:
                emb = item["embedding"]
                
                # Apply calibration if available
                if delta_v is not None:
                    calibrated_emb = emb + self.calibration_alpha * delta_v
                    calibrated_emb = calibrated_emb / np.linalg.norm(calibrated_emb)
                    sim = self.cosine_similarity(self.ref_embedding, calibrated_emb)
                else:
                    sim = self.cosine_similarity(self.ref_embedding, emb)
                    
                is_match = sim >= self.verify_threshold
                item_result = {
                    "filename": item["filename"],
                    "path": item["path"],
                    "similarity": sim,
                    "is_match": is_match
                }
                verified_results[folder_name].append(item_result)
                
                if is_match:
                    matches_count += 1
                    # Copy to matches folder
                    shutil.copy2(item["path"], os.path.join(match_dest_dir, item["filename"]))
                    
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
