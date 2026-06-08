import os
import sys
import logging
import json
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

# Add src directory to python path
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from qdrant_face_client import QdrantFaceClient
from minio_face_client import MinioFaceClient
from clustering import FaceClustering
from utils.color_normalizer import FaceImageNormalizer
from face_id_verifier import FaceIDVerifier

# Load environment variables
load_dotenv()

# Configure logging to both console and a log file in output directory
output_dir = os.getenv("OUTPUT_DATA_DIR", "data")
os.makedirs(output_dir, exist_ok=True)

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(output_dir, "clustering.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger("FaceClustering.Main")

class FaceIDNormalizeApp:
    """
    Unified Application class that orchestrates:
    1. Fetching face embeddings from Qdrant.
    2. Clustering them using HDBSCAN.
    3. Downloading the images from MinIO.
    4. Normalizing color space & color distribution of the images.
    5. Performing local alignment, re-embedding, and domain shift calibrated FaceID verification.
    """
    def __init__(self):
        logger.info("Initializing FaceID Normalize Application...")
        
        # Load parameters
        self.min_cluster_size = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "2"))
        self.min_samples = int(os.getenv("HDBSCAN_MIN_SAMPLES", "10"))
        self.metric = os.getenv("HDBSCAN_METRIC", "euclidean")
        self.output_dir = output_dir
        
        # Load normalization settings
        self.normalize_enabled = os.getenv("FACE_NORMALIZE_ENABLED", "0") == "1"
        self.normalize_method = os.getenv("FACE_NORMALIZE_METHOD", "reinhard")
        self.ref_image_path = os.getenv("FACE_REF_IMAGE_PATH", "data/_face_ID/duong_cropped.jpg")
        self.normalize_overwrite = os.getenv("FACE_NORMALIZE_OVERWRITE", "1") == "1"
        
        # Load verification settings
        self.verify_enabled = os.getenv("FACE_VERIFY_ENABLED", "0") == "1"
        self.verify_threshold = float(os.getenv("FACE_VERIFY_THRESHOLD", "0.65"))
        self.verify_calibration_enabled = os.getenv("FACE_VERIFY_CALIBRATION_ENABLED", "0") == "1"
        self.verify_calibration_alpha = float(os.getenv("FACE_VERIFY_CALIBRATION_ALPHA", "0.6"))
        
        # Initialize normalizer if enabled
        self.normalizer = None
        if self.normalize_enabled:
            try:
                self.normalizer = FaceImageNormalizer(
                    method=self.normalize_method,
                    reference_path=self.ref_image_path
                )
                logger.info("Face image normalizer initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize normalizer: {e}. Normalization disabled.")
                self.normalize_enabled = False
                
        # Initialize verifier if enabled
        self.verifier = None
        if self.verify_enabled:
            try:
                self.verifier = FaceIDVerifier(
                    ref_image_path=self.ref_image_path,
                    model_root=".",
                    verify_threshold=self.verify_threshold,
                    calibration_enabled=self.verify_calibration_enabled,
                    calibration_alpha=self.verify_calibration_alpha
                )
                logger.info("Local FaceID verifier initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize verifier: {e}. Verification disabled.")
                self.verify_enabled = False
        
        # Initialize clients
        self.qdrant_client = QdrantFaceClient()
        self.minio_client = MinioFaceClient()
        self.clustering_engine = FaceClustering(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric
        )

    def run(self):
        """Execute the entire pipeline."""
        start_time = datetime.now()
        logger.info(f"Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1. Fetch points from Qdrant
            points = self.qdrant_client.fetch_all_face_vectors()
            if not points:
                logger.warning("No face embeddings found in Qdrant database. Exiting pipeline.")
                return
            
            # 2. Extract vectors and keep map of index -> point metadata
            embeddings = []
            valid_points = []
            
            for point in points:
                payload = point.get("payload", {})
                minio_url = payload.get("minio_url")
                if not minio_url:
                    logger.warning(f"Point ID {point['id']} has no 'minio_url' in payload. Skipping.")
                    continue
                embeddings.append(point["vector"])
                valid_points.append(point)
                
            if not valid_points:
                logger.warning("No valid points with 'minio_url' found. Exiting pipeline.")
                return
                
            logger.info(f"Prepared {len(valid_points)} embeddings for clustering.")
            
            # 3. Perform clustering
            labels, probabilities = self.clustering_engine.run(embeddings)
            
            # 4. Process cluster results and organize files
            logger.info("Organizing images into cluster directories...")
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                clusters[int(label)].append({
                    "point_id": valid_points[idx]["id"],
                    "minio_url": valid_points[idx]["payload"]["minio_url"],
                    "payload": valid_points[idx]["payload"],
                    "probability": float(probabilities[idx]),
                    "vector": valid_points[idx]["vector"]
                })
            
            # Create directories and download files
            stats = {}
            total_downloaded = 0
            total_failed = 0
            
            for label, cluster_points in sorted(clusters.items()):
                cluster_dir_name = "noise" if label == -1 else f"cluster_{label}"
                logger.info(f"Processing {cluster_dir_name} ({len(cluster_points)} images)...")
                
                cluster_path = os.path.join(self.output_dir, cluster_dir_name)
                os.makedirs(cluster_path, exist_ok=True)
                
                stats[cluster_dir_name] = {
                    "count": len(cluster_points),
                    "items": []
                }
                
                for item in cluster_points:
                    minio_url = item["minio_url"]
                    _, object_name = self.minio_client.parse_minio_url(minio_url)
                    filename = object_name.split('/')[-1]
                    
                    short_id = str(item["point_id"])[:8]
                    dest_filename = f"{short_id}_{filename}"
                    dest_path = os.path.join(cluster_path, dest_filename)
                    
                    # Download image
                    success = self.minio_client.download_face_image(minio_url, dest_path)
                    if success:
                        total_downloaded += 1
                        
                        # Apply color normalization if enabled
                        if self.normalize_enabled and self.normalizer:
                            if self.normalize_overwrite:
                                if self.normalizer.transform_file(dest_path, dest_path):
                                    logger.info(f"Normalized and overwrote image: {dest_path}")
                            else:
                                norm_filename = f"{short_id}_normalized_{filename}"
                                norm_dest_path = os.path.join(cluster_path, norm_filename)
                                if self.normalizer.transform_file(dest_path, norm_dest_path):
                                    logger.info(f"Normalized image saved to: {norm_dest_path}")
                                    dest_path = norm_dest_path
                                    
                        item["local_path"] = dest_path
                        stats[cluster_dir_name]["items"].append(item)
                    else:
                        total_failed += 1
            
            # 5. Perform Face ID verification if enabled
            verification_stats = None
            if self.verify_enabled and self.verifier:
                # Prepare verification dict: folder_name -> list of items
                verification_items = {k: v["items"] for k, v in stats.items()}
                verification_stats = self.verifier.verify_vectors(verification_items, self.output_dir)
                
            # 6. Generate final reports
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self._generate_report(stats, verification_stats, duration, total_downloaded, total_failed)
            
            logger.info("==========================================================")
            logger.info(f"Pipeline finished successfully in {duration:.2f} seconds.")
            logger.info(f"Total downloaded: {total_downloaded}")
            logger.info(f"Total failed: {total_failed}")
            logger.info(f"Clusters found: {len(clusters) - (1 if -1 in clusters else 0)}")
            if verification_stats:
                logger.info(f"FaceID Verification Matches: {verification_stats['matches_count']}")
                logger.info(f"Matched images copied to: {verification_stats['match_dest_dir']}")
            logger.info("==========================================================")
            
        except Exception as e:
            logger.error(f"Critical error in main pipeline: {e}", exc_info=True)
            raise

    def _generate_report(
        self, 
        stats: dict, 
        verification_stats: Optional[dict], 
        duration: float, 
        downloaded: int, 
        failed: int
    ):
        """Generate comprehensive json and markdown reports."""
        report_path = os.path.join(self.output_dir, "clustering_report.md")
        json_path = os.path.join(self.output_dir, "clustering_report.json")
        
        # Save JSON
        combined_report = {
            "clustering_stats": stats,
            "verification_stats": verification_stats,
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "downloaded": downloaded,
                "failed": failed
            }
        }
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(combined_report, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write JSON report: {e}")

        # Save Markdown
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# Báo Cáo Kết Quả Phân Cụm & Đối Khớp FaceID\n\n")
                f.write(f"- **Thời gian chạy**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Thời gian xử lý**: {duration:.2f} giây\n")
                f.write(f"- **Tổng số ảnh tải thành công**: {downloaded}\n")
                f.write(f"- **Số ảnh lỗi**: {failed}\n")
                f.write(f"- **Cấu hình HDBSCAN**:\n")
                f.write(f"  - `min_cluster_size`: {self.min_cluster_size}\n")
                f.write(f"  - `min_samples`: {self.min_samples}\n")
                f.write(f"  - `metric`: {self.metric}\n\n")
                
                # FaceID Verification section
                if verification_stats:
                    f.write("## 1. Kết Quả Xác Thực FaceID & Hiệu Chỉnh Miền\n\n")
                    f.write(f"- **Ảnh chuẩn sử dụng**: `{self.ref_image_path}`\n")
                    f.write(f"- **Cụm đại diện nhận diện tự động**: `{verification_stats['target_folder']}`\n")
                    f.write(f"- **Độ dài Vector dịch chuyển miền ($\Delta v$ norm)**: {verification_stats['delta_v_norm']:.4f}\n")
                    f.write(f"- **Hệ số hiệu chỉnh tịnh tiến ($\alpha$)**: {self.verify_calibration_alpha}\n")
                    f.write(f"- **Ngưỡng nhận diện đối khớp**: {self.verify_threshold}\n")
                    f.write(f"- **Tổng số ảnh trùng khớp tìm thấy**: **{verification_stats['matches_count']}**\n")
                    f.write(f"- **Thư mục lưu ảnh khớp**: `data/matches/{os.path.splitext(os.path.basename(self.ref_image_path))[0]}/`\n\n")
                    
                    # Top 10 matches table
                    f.write("### Top 10 Ảnh Có Độ Tương Đồng Cao Nhất Sau Hiệu Chỉnh\n\n")
                    f.write("| # | Cụm | Tên File | Độ Tương Đồng | Trạng Thái |\n")
                    f.write("| :---: | :---: | :--- | :---: | :---: |")
                    
                    all_matched_items = []
                    for folder, items in verification_stats["results"].items():
                        for item in items:
                            all_matched_items.append({
                                "folder": folder,
                                "filename": item["filename"],
                                "similarity": item["similarity"],
                                "is_match": item["is_match"]
                            })
                    all_matched_items = sorted(all_matched_items, key=lambda x: x["similarity"], reverse=True)
                    
                    for idx, item in enumerate(all_matched_items[:10]):
                        status = "✅ KHỚP (Duong)" if item["is_match"] else "❌ KHÔNG KHỚP"
                        f.write(f"\n| {idx+1} | `{item['folder']}` | `{item['filename']}` | {item['similarity']:.4f} | {status} |")
                    f.write("\n\n")
                
                # Clustering statistics section
                f.write("## 2. Thống Kê Các Cụm Phân Nhóm (Clustering)\n\n")
                f.write("| Tên Thư Mục | Số Lượng Ảnh | Trạng Thái |")
                if verification_stats:
                    f.write(" Số Ảnh Khớp |")
                f.write("\n| :--- | :---: | :--- |")
                if verification_stats:
                    f.write(" :---: |")
                f.write("\n")
                
                total_clusters = 0
                for folder_name, info in sorted(stats.items()):
                    status = "Nhiễu / Outliers" if folder_name == "noise" else "Cụm hợp lệ"
                    if folder_name != "noise":
                        total_clusters += 1
                        
                    f.write(f"| `{folder_name}` | {info['count']} | {status} |")
                    if verification_stats:
                        folder_matches = sum(1 for x in verification_stats["results"].get(folder_name, []) if x["is_match"])
                        f.write(f" {folder_matches} |")
                    f.write("\n")
                
                f.write(f"\n- **Tổng số cụm phân cụm tìm thấy**: {total_clusters}\n\n")
                
                # Detail section
                f.write("## 3. Chi Tiết Các Ảnh Trong Cụm\n\n")
                for folder_name, info in sorted(stats.items()):
                    f.write(f"### {folder_name.upper()} ({info['count']} ảnh)\n\n")
                    f.write("| Point ID | Tên File Local |")
                    if verification_stats:
                        f.write(" Độ Tương Đồng (Đã hiệu chỉnh) | Trạng Thái |")
                    f.write(" MinIO URL |\n")
                    f.write("| :--- | :--- |")
                    if verification_stats:
                        f.write(" :---: | :---: |")
                    f.write(" :--- |\n")
                    
                    # Create lookup for verification similarity
                    ver_map = {}
                    if verification_stats:
                        ver_map = {x["filename"]: x for x in verification_stats["results"].get(folder_name, [])}
                        
                    for item in info["items"]:
                        filename = os.path.basename(item["local_path"])
                        f.write(f"| `{item['point_id']}` | `{filename}` |")
                        if verification_stats:
                            res = ver_map.get(filename, {"similarity": 0.0, "is_match": False})
                            status = "Match" if res["is_match"] else "No Match"
                            f.write(f" {res['similarity']:.4f} | {status} |")
                        f.write(f" [Link]({item['minio_url']}) |\n")
                    f.write("\n")
                    
            logger.info(f"Markdown report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write markdown report: {e}")

if __name__ == "__main__":
    try:
        app = FaceIDNormalizeApp()
        app.run()
    except Exception as e:
        logger.critical(f"Application terminated abnormally: {e}")
        sys.exit(1)
