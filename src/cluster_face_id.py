import os
import sys
import glob
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app import FaceAnalysis

# Configure logger
logger = logging.getLogger("FaceClustering.LocalMatcher")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class LocalFaceIDMatcher:
    """
    Class to perform local FaceID alignment, re-embedding, and matching 
    against a reference image for all clustered folders.
    """
    def __init__(self, ref_image_path: str, model_root: str = "."):
        self.ref_image_path = ref_image_path
        self.model_root = model_root
        self.app: Optional[FaceAnalysis] = None
        self.ref_embedding: Optional[np.ndarray] = None
        
        # Initialize models
        self._init_models()
        # Extract reference embedding
        self._extract_reference_embedding()

    def _init_models(self) -> None:
        """Initialize InsightFace models."""
        try:
            logger.info("Initializing FaceAnalysis for local alignment & embedding...")
            self.app = FaceAnalysis(name='buffalo_m', root=self.model_root)
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("FaceAnalysis initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FaceAnalysis: {e}", exc_info=True)
            raise

    def _extract_reference_embedding(self) -> None:
        """Extract embedding from reference image."""
        try:
            logger.info(f"Loading reference image from: {self.ref_image_path}")
            img = cv2.imread(self.ref_image_path)
            if img is None:
                raise FileNotFoundError(f"Could not read reference image: {self.ref_image_path}")
                
            faces = self.app.get(img)
            if not faces:
                raise ValueError("No face detected in reference image! Alignment failed.")
                
            # Pick largest face
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            self.ref_embedding = faces[0].embedding
            logger.info("Reference face embedding extracted and aligned successfully.")
        except Exception as e:
            logger.error(f"Error extracting reference embedding: {e}", exc_info=True)
            raise

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def process_image(self, file_path: str) -> Optional[float]:
        """
        Applies padding to image, extracts embedding using local model,
        and computes similarity against the reference.
        """
        try:
            img = cv2.imread(file_path)
            if img is None:
                logger.warning(f"Could not read image: {file_path}")
                return None
                
            # Apply 50px border padding to assist SCRFD detector in finding tight crops
            padded_img = cv2.copyMakeBorder(
                img, 50, 50, 50, 50, 
                borderType=cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]
            )
            
            # Detect, align and embed
            faces = self.app.get(padded_img)
            if not faces:
                logger.debug(f"Face detection failed even with padding for: {file_path}")
                return None
                
            # Pick largest face
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            emb = faces[0].embedding
            
            # Compute similarity
            return self.cosine_similarity(self.ref_embedding, emb)
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return None

    def evaluate_clusters(self, base_data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scans all folders in base_data_dir, re-embeds images,
        and matches them against reference.
        """
        folders = ["cluster_0", "cluster_1", "noise"]
        results = {}
        
        for folder in folders:
            folder_path = os.path.join(base_data_dir, folder)
            if not os.path.exists(folder_path):
                logger.warning(f"Folder path not found: {folder_path}")
                continue
                
            jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
            logger.info(f"Processing {len(jpg_files)} images in folder: '{folder}'...")
            
            folder_results = []
            for path in jpg_files:
                similarity = self.process_image(path)
                if similarity is not None:
                    folder_results.append({
                        "filename": os.path.basename(path),
                        "path": path,
                        "similarity": similarity
                    })
                    
            # Sort by similarity descending
            folder_results = sorted(folder_results, key=lambda x: x["similarity"], reverse=True)
            results[folder] = folder_results
            
        return results

    def save_report(self, results: Dict[str, List[Dict[str, Any]]], output_path: str) -> None:
        """Generates a markdown report summarizing findings."""
        try:
            logger.info(f"Generating FaceID report to: {output_path}")
            
            # Prepare markdown content
            md = []
            md.append("# Báo cáo Đối khớp FaceID Cục bộ (Local Alignment & Re-embedding)\n")
            md.append(f"- **Ảnh chuẩn sử dụng**: `{self.ref_image_path}`")
            md.append("- **Thuật toán dò tìm & căn chỉnh**: SCRFD (InsightFace) + 50px Padding Border")
            md.append("- **Mô hình trích xuất đặc trưng**: `buffalo_m` (w600k_r50.onnx)\n")
            
            md.append("## 1. Tóm tắt kết quả theo cụm\n")
            md.append("| Cụm thư mục | Tổng số ảnh | Số ảnh Align thành công | Similarity trung bình | Similarity lớn nhất |")
            md.append("| :--- | :---: | :---: | :---: | :---: |")
            
            for folder, items in results.items():
                if not items:
                    md.append(f"| {folder} | 0 | 0 | 0.0000 | 0.0000 |")
                    continue
                sims = [x["similarity"] for x in items]
                avg_sim = np.mean(sims)
                max_sim = np.max(sims)
                md.append(f"| {folder} | {len(items)} | {len(items)} | {avg_sim:.4f} | {max_sim:.4f} |")
            
            md.append("\n## 2. Top 10 ảnh có độ tương đồng cao nhất\n")
            md.append("| # | Cụm | Tên file ảnh | Cosine Similarity | Trạng thái (Ngưỡng 0.65) |")
            md.append("| :---: | :---: | :--- | :---: | :---: |")
            
            # Flatten and sort all matches
            all_matches = []
            for folder, items in results.items():
                for item in items:
                    all_matches.append({
                        "folder": folder,
                        "filename": item["filename"],
                        "similarity": item["similarity"]
                    })
            all_matches = sorted(all_matches, key=lambda x: x["similarity"], reverse=True)
            
            for idx, item in enumerate(all_matches[:10]):
                status = "✅ ĐẠT" if item["similarity"] >= 0.65 else "❌ KHÔNG ĐẠT"
                md.append(f"| {idx+1} | `{item['folder']}` | `{item['filename']}` | {item['similarity']:.4f} | {status} |")
                
            # Write file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md))
                
            logger.info("Report generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

def main():
    ref_path = "data/_face_ID/duong_cropped.jpg"
    base_dir = "data"
    report_path = "data/local_face_id_report.md"
    
    matcher = LocalFaceIDMatcher(ref_image_path=ref_path)
    results = matcher.evaluate_clusters(base_dir)
    matcher.save_report(results, report_path)

if __name__ == "__main__":
    main()
