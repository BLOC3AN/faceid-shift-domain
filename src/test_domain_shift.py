import os
import sys
import glob
import cv2
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceClustering.DomainShift")

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def main():
    ref_path = "data/_face_ID/duong_cropped.jpg"
    cluster_1_dir = "data/cluster_1"
    cluster_0_dir = "data/cluster_0"
    
    app = FaceAnalysis(name='buffalo_m', root='.')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # 1. Extract reference embedding
    ref_img = cv2.imread(ref_path)
    ref_faces = app.get(ref_img)
    ref_emb = ref_faces[0].embedding
    # Normalize reference embedding
    ref_emb = ref_emb / np.linalg.norm(ref_emb)
    
    # 2. Extract embeddings for all cluster_1 images (using padding alignment)
    print("Extracting embeddings for cluster_1 (Duong)...")
    c1_embs = []
    for p in glob.glob(os.path.join(cluster_1_dir, "*.jpg")):
        img = cv2.imread(p)
        if img is None:
            continue
        padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        faces = app.get(padded)
        if faces:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            emb = faces[0].embedding
            c1_embs.append(emb / np.linalg.norm(emb))
            
    if not c1_embs:
        print("Error: No faces detected in cluster_1")
        return
        
    # 3. Calculate Centroid of cluster_1 and the translation vector
    centroid_c1 = np.mean(c1_embs, axis=0)
    centroid_c1 = centroid_c1 / np.linalg.norm(centroid_c1)
    
    delta_v = ref_emb - centroid_c1
    print(f"Centroid calculated. Translation vector delta_v norm: {np.linalg.norm(delta_v):.4f}")
    
    # 4. Extract embeddings for cluster_0 (Other person) for validation
    print("Extracting embeddings for cluster_0 (Other person)...")
    c0_embs = []
    for p in glob.glob(os.path.join(cluster_0_dir, "*.jpg"))[:30]: # Sample 30 images
        img = cv2.imread(p)
        if img is None:
            continue
        padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        faces = app.get(padded)
        if faces:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            emb = faces[0].embedding
            c0_embs.append(emb / np.linalg.norm(emb))
            
    # 5. Evaluate different translation strengths alpha
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("\n" + "="*80)
    print("EFFECT OF DOMAIN SHIFT TRANSLATION ON COSINE SIMILARITY")
    print("="*80)
    print(f"{'Alpha':6s} | {'C1 (Duong) Avg Sim':20s} | {'C1 (Duong) Max Sim':20s} | {'C0 (Other) Avg Sim':20s}")
    print("-" * 80)
    
    for alpha in alphas:
        # Evaluate C1 (Duong)
        sims_c1 = []
        for emb in c1_embs:
            # Apply translation
            calibrated_emb = emb + alpha * delta_v
            calibrated_emb = calibrated_emb / np.linalg.norm(calibrated_emb)
            sims_c1.append(cosine_similarity(ref_emb, calibrated_emb))
            
        # Evaluate C0 (Other)
        sims_c0 = []
        for emb in c0_embs:
            calibrated_emb = emb + alpha * delta_v
            calibrated_emb = calibrated_emb / np.linalg.norm(calibrated_emb)
            sims_c0.append(cosine_similarity(ref_emb, calibrated_emb))
            
        print(f"{alpha:6.1f} | {np.mean(sims_c1):20.4f} | {np.max(sims_c1):20.4f} | {np.mean(sims_c0):20.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
