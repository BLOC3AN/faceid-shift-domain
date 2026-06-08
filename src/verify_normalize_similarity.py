import os
import sys
import glob
import cv2
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FaceClustering.VerifyNormalize")

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def main():
    ref_path = "data/_face_ID/duong.jpg"
    if not os.path.exists(ref_path):
        logger.error(f"Reference image not found at '{ref_path}'")
        return
        
    logger.info("Initializing FaceAnalysis...")
    app = FaceAnalysis(name='buffalo_m', root='.')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Extract reference embedding
    img = cv2.imread(ref_path)
    faces = app.get(img)
    if not faces:
        logger.error("No face detected in reference image!")
        return
    duong_embedding = faces[0].embedding
    
    # We will sample 20 images from cluster_1 and 20 from cluster_0 (now normalized on disk)
    # to compare their new embeddings with duong.jpg.
    for cluster_name in ["cluster_0", "cluster_1"]:
        cluster_dir = os.path.join("data", cluster_name)
        if not os.path.exists(cluster_dir):
            continue
            
        jpg_files = glob.glob(os.path.join(cluster_dir, "*.jpg"))
        if not jpg_files:
            logger.warning(f"No files found in {cluster_name}")
            continue
            
        logger.info(f"Processing {min(30, len(jpg_files))} normalized images from {cluster_name}...")
        
        similarities = []
        for file_path in jpg_files[:30]:
            test_img = cv2.imread(file_path)
            if test_img is None:
                continue
            test_faces = app.get(test_img)
            if test_faces:
                # Get the largest face
                test_faces = sorted(test_faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                sim = cosine_similarity(duong_embedding, test_faces[0].embedding)
                similarities.append(sim)
                
        if similarities:
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
            min_sim = np.min(similarities)
            logger.info(
                f"New similarity for {cluster_name} (Normalized):"
                f"  - Average: {avg_sim:.4f}"
                f"  - Max: {max_sim:.4f}"
                f"  - Min: {min_sim:.4f}"
            )

if __name__ == "__main__":
    main()
