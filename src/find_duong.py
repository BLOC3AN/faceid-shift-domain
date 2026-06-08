import os
import sys
import json
import cv2
import logging
import numpy as np

# Add project root and src to python path
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from qdrant_face_client import QdrantFaceClient
from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FaceClustering.FindDuong")

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def main():
    ref_path = "data/_face_ID/duong.jpg"
    report_json_path = "data/clustering_report.json"
    
    if not os.path.exists(ref_path):
        logger.error(f"Reference image not found at '{ref_path}'")
        sys.exit(1)
        
    if not os.path.exists(report_json_path):
        logger.error(f"Clustering report not found at '{report_json_path}'. Please run pipeline first.")
        sys.exit(1)
        
    # 1. Load clustering assignment from report
    logger.info(f"Loading clustering report from {report_json_path}...")
    with open(report_json_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
        
    # Map point_id (string) to cluster name
    point_to_cluster = {}
    for cluster_name, info in report_data.items():
        for item in info.get("items", []):
            point_id = str(item["point_id"])
            point_to_cluster[point_id] = cluster_name
            
    logger.info(f"Loaded {len(point_to_cluster)} point mappings from report.")
    
    # 2. Extract embedding of the standard image (Duong)
    logger.info(f"Extracting embedding of standard image: {ref_path}")
    img = cv2.imread(ref_path)
    if img is None:
        logger.error("Failed to read reference image.")
        sys.exit(1)
        
    # Initialize FaceAnalysis
    app = FaceAnalysis(name='buffalo_m', root='.')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    faces = app.get(img)
    if len(faces) == 0:
        logger.error("No face detected in the reference image!")
        sys.exit(1)
    elif len(faces) > 1:
        logger.warning(f"Multiple faces ({len(faces)}) detected in reference image. Using the largest face.")
        # Sort by box size descending and pick first
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        
    duong_embedding = faces[0].embedding
    logger.info(f"Extracted embedding for Duong. Shape: {duong_embedding.shape}")
    
    # 3. Fetch all embeddings from Qdrant
    logger.info("Connecting to Qdrant to fetch vectors...")
    qdrant_client = QdrantFaceClient()
    qdrant_points = qdrant_client.fetch_all_face_vectors()
    
    if not qdrant_points:
        logger.error("No vectors fetched from Qdrant.")
        sys.exit(1)
        
    # 4. Group vectors by cluster
    cluster_similarities = {}
    
    for point in qdrant_points:
        point_id = str(point["id"])
        vector = np.array(point["vector"], dtype=np.float32)
        
        # Check if this point belongs to one of the clusters in our report
        cluster_name = point_to_cluster.get(point_id)
        if cluster_name:
            if cluster_name not in cluster_similarities:
                cluster_similarities[cluster_name] = []
                
            # Compute cosine similarity
            sim = cosine_similarity(duong_embedding, vector)
            cluster_similarities[cluster_name].append({
                "point_id": point_id,
                "similarity": sim,
                "minio_url": point["payload"].get("minio_url", "N/A")
            })
            
    # 5. Compute statistics and print results
    logger.info("\n" + "="*60 + "\nRESULTS: SIMILARITY STATISTICS FOR EACH CLUSTER\n" + "="*60)
    
    results = []
    
    # Similarity threshold to declare a match (usually 0.5-0.6 is a match for buffalo_m)
    MATCH_THRESHOLD = 0.55
    
    for cluster_name, sim_items in sorted(cluster_similarities.items()):
        sims = [x["similarity"] for x in sim_items]
        if not sims:
            continue
            
        avg_sim = np.mean(sims)
        max_sim = np.max(sims)
        min_sim = np.min(sims)
        std_sim = np.std(sims)
        
        matches = [x for x in sims if x >= MATCH_THRESHOLD]
        match_ratio = len(matches) / len(sims)
        
        logger.info(
            f"Cluster: {cluster_name}\n"
            f"  - Count: {len(sims)} faces\n"
            f"  - Average Cosine Similarity: {avg_sim:.4f}\n"
            f"  - Max Similarity: {max_sim:.4f}\n"
            f"  - Min Similarity: {min_sim:.4f}\n"
            f"  - Standard Deviation: {std_sim:.4f}\n"
            f"  - Match Count (>= {MATCH_THRESHOLD}): {len(matches)} ({match_ratio*100:.2f}%)\n"
        )
        
        results.append({
            "cluster": cluster_name,
            "avg_sim": avg_sim,
            "match_ratio": match_ratio,
            "count": len(sims)
        })
        
    logger.info("="*60)
    
    # 6. Make final conclusion
    if results:
        # Sort clusters by average similarity descending
        results = sorted(results, key=lambda x: x["avg_sim"], reverse=True)
        best_match = results[0]
        
        if best_match["avg_sim"] >= MATCH_THRESHOLD:
            print(f"\n[CONCLUSION] Người tên Duong nằm ở cụm: **{best_match['cluster']}**")
            print(f"  - Độ tương đồng trung bình: {best_match['avg_sim']:.4f}")
            print(f"  - Tỉ lệ khớp khuôn mặt: {best_match['match_ratio']*100:.2f}% ({best_match['count']} ảnh)")
            print(f"  - Cụm đối thủ còn lại có độ tương đồng trung bình: "
                  f"{', '.join([f'{r[chr(99)+chr(108)+chr(117)+chr(115)+chr(116)+chr(101)+chr(114)]}: {r[chr(97)+chr(118)+chr(103)+_][0] if isinstance(r[chr(97)+chr(118)+chr(103)+_], list) else r[chr(97)+chr(118)+chr(103)+chr(115)+chr(105)+chr(109)]:.4f}' for r in results[1:]]) if len(results) > 1 else 'Không có'}")
        else:
            print("\n[CONCLUSION] Không tìm thấy cụm nào khớp đáng tin cậy với người tên Duong.")
            print(f"  - Cụm gần nhất là {best_match['cluster']} chỉ đạt độ tương đồng trung bình {best_match['avg_sim']:.4f}")

if __name__ == "__main__":
    main()
