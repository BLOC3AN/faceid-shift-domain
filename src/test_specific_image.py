import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app import FaceAnalysis
from qdrant_face_client import QdrantFaceClient

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def main():
    ref_path = "data/_face_ID/duong.jpg"
    target_path = "data/cluster_1/6627b526_face_20260605_101942_id1_conf0.99.jpg"
    
    if not os.path.exists(ref_path):
        print(f"Error: Reference path {ref_path} not found")
        return
    if not os.path.exists(target_path):
        print(f"Error: Target path {target_path} not found")
        return
        
    print("Initializing FaceAnalysis...")
    app = FaceAnalysis(name='buffalo_m', root='.')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Extract reference embedding
    print("Extracting embedding for reference image...")
    ref_img = cv2.imread(ref_path)
    ref_faces = app.get(ref_img)
    if not ref_faces:
        print("Error: No face detected in reference image")
        return
    ref_embedding = ref_faces[0].embedding
    
    # Extract target image embedding (Normalized on disk)
    print("Extracting embedding for target image from disk (CLAHE Normalized)...")
    target_img = cv2.imread(target_path)
    target_faces = app.get(target_img)
    if not target_faces:
        print("Error: No face detected in target image")
        return
    target_embedding_normalized = target_faces[0].embedding
    
    # Compute similarity for normalized image
    sim_normalized = cosine_similarity(ref_embedding, target_embedding_normalized)
    
    # Fetch original vector from Qdrant
    print("Fetching original embedding from Qdrant...")
    qdrant_client = QdrantFaceClient()
    qdrant_points = qdrant_client.fetch_all_face_vectors()
    
    # Find matching point ID '6627b526'
    orig_embedding = None
    for point in qdrant_points:
        point_id = str(point["id"])
        if point_id.startswith("6627b526"):
            orig_embedding = np.array(point["vector"], dtype=np.float32)
            break
            
    sim_original = None
    if orig_embedding is not None:
        sim_original = cosine_similarity(ref_embedding, orig_embedding)
    
    print("\nRESULTS FOR SPECIFIC IMAGE:")
    print("="*60)
    print(f"Target Image: {os.path.basename(target_path)}")
    if sim_original is not None:
        print(f"- Original Similarity (Qdrant - Un-normalized): {sim_original:.4f}")
    else:
        print("- Original Similarity (Qdrant): Not found in DB")
    print(f"- Current Similarity (Local Disk - CLAHE Normalized): {sim_normalized:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
