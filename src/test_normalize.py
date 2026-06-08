import os
import sys
import glob
import cv2
import logging
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.color_normalizer import FaceImageNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FaceClustering.TestNormalize")

def add_label_to_image(img: np.ndarray, text: str) -> np.ndarray:
    """
    Draw a background rectangle and put text on top of the image for labeling.
    """
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size[0], text_size[1]
    
    # Draw background box at the bottom
    cv2.rectangle(
        img_copy, 
        (0, h - text_h - 10), 
        (w, h), 
        (0, 0, 0), 
        -1
    )
    
    # Put text
    text_x = (w - text_w) // 2
    text_y = h - 5
    cv2.putText(
        img_copy, 
        text, 
        (text_x, text_y), 
        font, 
        font_scale, 
        (255, 255, 255), 
        thickness, 
        cv2.LINE_AA
    )
    return img_copy

def run_comparison():
    ref_path = "data/_face_ID/duong.jpg"
    if not os.path.exists(ref_path):
        logger.error(f"Reference image not found at '{ref_path}'")
        return
        
    # Find some sample face images in clusters
    cluster_dirs = ["data/cluster_0", "data/cluster_1", "data/noise"]
    sample_images = []
    
    for c_dir in cluster_dirs:
        if os.path.exists(c_dir):
            jpgs = glob.glob(os.path.join(c_dir, "*.jpg"))
            if jpgs:
                # Add up to 2 samples from each directory
                sample_images.extend(jpgs[:2])
                
    if not sample_images:
        logger.error("No sample images found in cluster directories to test.")
        return
        
    logger.info(f"Found {len(sample_images)} samples to test.")
    
    # Initialize normalizers for each method
    normalizers = {
        "Reinhard": FaceImageNormalizer(method="reinhard", reference_path=ref_path),
        "Histogram": FaceImageNormalizer(method="histogram", reference_path=ref_path),
        "CLAHE": FaceImageNormalizer(method="clahe", reference_path=ref_path),
        "Hybrid": FaceImageNormalizer(method="hybrid", reference_path=ref_path),
    }
    
    # Read reference image and resize to a standard display size (e.g., 150x150)
    ref_img = cv2.imread(ref_path)
    display_size = (150, 150)
    ref_img_resized = cv2.resize(ref_img, display_size)
    ref_labeled = add_label_to_image(ref_img_resized, "Reference")
    
    # Process each sample image
    for idx, sample_path in enumerate(sample_images):
        logger.info(f"Processing sample {idx+1}: {sample_path}")
        
        src_img = cv2.imread(sample_path)
        if src_img is None:
            logger.warning(f"Could not read sample image: {sample_path}")
            continue
            
        # Resize source image to match display size
        src_resized = cv2.resize(src_img, display_size)
        src_labeled = add_label_to_image(src_resized, "Original")
        
        row_images = [ref_labeled, src_labeled]
        
        # Apply each normalization method
        for name, normalizer in normalizers.items():
            try:
                # Transform the resized source image
                norm_img = normalizer.transform(src_resized)
                # Label the image
                norm_labeled = add_label_to_image(norm_img, name)
                row_images.append(norm_labeled)
            except Exception as e:
                logger.error(f"Error executing {name} on {sample_path}: {e}")
                # Placeholder in case of error
                error_img = src_resized.copy()
                cv2.putText(error_img, "ERROR", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                row_images.append(error_img)
                
        # Concatenate horizontally
        comparison_row = np.hstack(row_images)
        
        # Save comparison result
        output_filename = f"normalize_comparison_sample_{idx+1}.jpg"
        output_path = os.path.join("data", output_filename)
        cv2.imwrite(output_path, comparison_row)
        logger.info(f"Saved comparison row to {output_path}")

if __name__ == "__main__":
    run_comparison()
