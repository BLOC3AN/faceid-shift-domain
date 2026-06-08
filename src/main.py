import os
import sys
import logging
import json
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Add project root and src to python path if not already there
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
    Main Application class that orchestrates:
    1. Fetching face embeddings from Qdrant.
    2. Clustering them using HDBSCAN.
    3. Downloading the images from MinIO.
    4. Grouping them into folders under the data/ directory.
    """
    def __init__(self):
        logger.info("Initializing FaceID Normalize Application...")
        
        # Load parameters
        self.min_cluster_size = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "3"))
        self.min_samples = int(os.getenv("HDBSCAN_MIN_SAMPLES", "1"))
        self.metric = os.getenv("HDBSCAN_METRIC", "euclidean")
        self.output_dir = output_dir
        
        # Load normalization settings
        self.normalize_enabled = os.getenv("FACE_NORMALIZE_ENABLED", "0") == "1"
        self.normalize_method = os.getenv("FACE_NORMALIZE_METHOD", "reinhard")
        self.ref_image_path = os.getenv("FACE_REF_IMAGE_PATH", "data/_face_ID/duong.jpg")
        self.normalize_overwrite = os.getenv("FACE_NORMALIZE_OVERWRITE", "1") == "1"
        
        # Initialize normalizer if enabled
        self.normalizer = None
        if self.normalize_enabled:
            try:
                self.normalizer = FaceImageNormalizer(
                    method=self.normalize_method,
                    reference_path=self.ref_image_path
                )
                logger.info("Face normalization is enabled and initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize FaceImageNormalizer: {e}. Normalization will be disabled.")
                self.normalize_enabled = False
        
        # Initialize clients
        self.qdrant_client = QdrantFaceClient()
        self.minio_client = MinioFaceClient()
        self.clustering_engine = FaceClustering(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric
        )

    def run(self):
        """
        Execute the entire pipeline.
        """
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
                
                # We can only process points that have a valid minio_url
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
            
            # Group points by their cluster labels
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                clusters[int(label)].append({
                    "point_id": valid_points[idx]["id"],
                    "minio_url": valid_points[idx]["payload"]["minio_url"],
                    "payload": valid_points[idx]["payload"],
                    "probability": float(probabilities[idx])
                })
            
            # Create directories and download files
            stats = {}
            total_downloaded = 0
            total_failed = 0
            
            for label, cluster_points in sorted(clusters.items()):
                # -1 is noise
                if label == -1:
                    cluster_dir_name = "noise"
                    logger.info(f"Processing Noise/Outliers ({len(cluster_points)} images)...")
                else:
                    cluster_dir_name = f"cluster_{label}"
                    logger.info(f"Processing Cluster {label} ({len(cluster_points)} images)...")
                    
                cluster_path = os.path.join(self.output_dir, cluster_dir_name)
                os.makedirs(cluster_path, exist_ok=True)
                
                stats[cluster_dir_name] = {
                    "count": len(cluster_points),
                    "items": []
                }
                
                for item in cluster_points:
                    minio_url = item["minio_url"]
                    # Extract original filename
                    _, object_name = self.minio_client.parse_minio_url(minio_url)
                    filename = object_name.split('/')[-1]
                    
                    # To prevent name collision, prefix with Qdrant Point ID (first 8 chars)
                    short_id = str(item["point_id"])[:8]
                    dest_filename = f"{short_id}_{filename}"
                    dest_path = os.path.join(cluster_path, dest_filename)
                    
                    # Download
                    success = self.minio_client.download_face_image(minio_url, dest_path)
                    if success:
                        total_downloaded += 1
                        
                        # Apply normalization if enabled
                        if self.normalize_enabled and self.normalizer:
                            if self.normalize_overwrite:
                                # Overwrite the downloaded file
                                if self.normalizer.transform_file(dest_path, dest_path):
                                    logger.info(f"Normalized and overwrote image: {dest_path}")
                                else:
                                    logger.warning(f"Normalization failed to overwrite {dest_path}, keeping original.")
                            else:
                                # Save as a new file with suffix
                                norm_filename = f"{short_id}_normalized_{filename}"
                                norm_dest_path = os.path.join(cluster_path, norm_filename)
                                if self.normalizer.transform_file(dest_path, norm_dest_path):
                                    logger.info(f"Normalized image saved to: {norm_dest_path}")
                                    # Update local_path to the normalized path
                                    dest_path = norm_dest_path
                                else:
                                    logger.warning(f"Normalization failed for {dest_path}, keeping original.")
                                    
                        item["local_path"] = dest_path
                        stats[cluster_dir_name]["items"].append(item)
                    else:
                        total_failed += 1
                        
            # 5. Generate report
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self._generate_report(stats, duration, total_downloaded, total_failed)
            
            logger.info("==========================================================")
            logger.info(f"Pipeline finished successfully in {duration:.2f} seconds.")
            logger.info(f"Total images processed: {len(valid_points)}")
            logger.info(f"Total downloaded: {total_downloaded}")
            logger.info(f"Total failed: {total_failed}")
            logger.info(f"Clusters found: {len(clusters) - (1 if -1 in clusters else 0)}")
            logger.info(f"Report saved to: {os.path.join(self.output_dir, 'clustering_report.md')}")
            logger.info("==========================================================")
            
        except Exception as e:
            logger.error(f"Critical error in main pipeline: {e}", exc_info=True)
            raise

    def _generate_report(self, stats: dict, duration: float, downloaded: int, failed: int):
        """
        Generate a comprehensive markdown report of the clustering process.
        """
        report_path = os.path.join(self.output_dir, "clustering_report.md")
        json_path = os.path.join(self.output_dir, "clustering_report.json")
        
        # Save detailed JSON report
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write JSON report: {e}")

        # Save markdown report
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# Báo Cáo Kết Quả Phân Cụm Khuôn Mặt (HDBSCAN)\n\n")
                f.write(f"- **Thời gian chạy**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Thời gian xử lý**: {duration:.2f} giây\n")
                f.write(f"- **Tổng số ảnh tải thành công**: {downloaded}\n")
                f.write(f"- **Số ảnh lỗi**: {failed}\n")
                f.write(f"- **Cấu hình HDBSCAN**:\n")
                f.write(f"  - `min_cluster_size`: {self.min_cluster_size}\n")
                f.write(f"  - `min_samples`: {self.min_samples}\n")
                f.write(f"  - `metric`: {self.metric}\n\n")
                
                f.write("## Thống kê các Cụm (Clusters)\n\n")
                f.write("| Tên Thư Mục | Số Lượng Ảnh | Trạng Thái |\n")
                f.write("| :--- | :---: | :--- |\n")
                
                total_clusters = 0
                for folder_name, info in sorted(stats.items()):
                    status = "Nhiễu / Outliers" if folder_name == "noise" else "Cụm hợp lệ"
                    if folder_name != "noise":
                        total_clusters += 1
                    f.write(f"| `{folder_name}` | {info['count']} | {status} |\n")
                
                f.write(f"\n- **Tổng số cụm tìm thấy**: {total_clusters}\n\n")
                
                f.write("## Chi Tiết Các Ảnh Trong Cụm\n\n")
                for folder_name, info in sorted(stats.items()):
                    f.write(f"### {folder_name.upper()} ({info['count']} ảnh)\n\n")
                    f.write("| Point ID | Tên File Local | Xác Suất Phân Cụm | MinIO URL |\n")
                    f.write("| :--- | :--- | :---: | :--- |\n")
                    for item in info["items"]:
                        filename = os.path.basename(item["local_path"])
                        f.write(f"| `{item['point_id']}` | `{filename}` | {item['probability']:.4f} | [Link]({item['minio_url']}) |\n")
                    f.write("\n")
                    
        except Exception as e:
            logger.error(f"Failed to write markdown report: {e}")

if __name__ == "__main__":
    try:
        app = FaceIDNormalizeApp()
        app.run()
    except Exception as e:
        logger.critical(f"Application terminated abnormally: {e}")
        sys.exit(1)
