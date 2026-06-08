import os
import sys
import json
import cv2
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

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def main():
    ref_path = "data/_face_ID/duong.jpg"
    report_json_path = "data/clustering_report.json"
    
    with open(report_json_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
        
    point_to_info = {}
    for cluster_name, info in report_data.items():
        for item in info.get("items", []):
            point_id = str(item["point_id"])
            point_to_info[point_id] = {
                "cluster": cluster_name,
                "local_path": item["local_path"],
                "minio_url": item["minio_url"]
            }
            
    # Extract reference embedding
    img = cv2.imread(ref_path)
    app = FaceAnalysis(name='buffalo_m', root='.')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(img)
    duong_embedding = faces[0].embedding
    
    # Fetch from Qdrant
    qdrant_client = QdrantFaceClient()
    qdrant_points = qdrant_client.fetch_all_face_vectors()
    
    matches = []
    for point in qdrant_points:
        point_id = str(point["id"])
        if point_id in point_to_info and point_to_info[point_id]["cluster"] == "cluster_1":
            vector = np.array(point["vector"], dtype=np.float32)
            sim = cosine_similarity(duong_embedding, vector)
            matches.append({
                "point_id": point_id,
                "similarity": sim,
                "local_path": point_to_info[point_id]["local_path"],
                "minio_url": point_to_info[point_id]["minio_url"]
            })
            
    # Sort matches descending
    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
    
    print("\nTOP 5 BEST MATCHES FOR DUONG IN CLUSTER_1:")
    print("="*80)
    for idx, match in enumerate(matches[:5]):
        filename = os.path.basename(match["local_path"])
        print(f"{idx+1}. File: {filename}")
        print(f"   Similarity: {match['similarity']:.4f}")
        print(f"   MinIO URL: {match['minio_url']}")
        print(f"   Point ID: {match['point_id']}")
        print("-"*80)

if __name__ == "__main__":
    main()
