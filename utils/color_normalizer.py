import os
import cv2
import logging
import numpy as np
from typing import Optional, Tuple, Union
from skimage.exposure import match_histograms
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("FaceClustering.ColorNormalizer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class FaceImageNormalizer:
    """
    OOP class to handle color space and distribution normalization for face images.
    Supports Reinhard Color Transfer, Histogram Matching, CLAHE, and a Hybrid method.
    """
    def __init__(self, method: Optional[str] = None, reference_path: Optional[str] = None):
        """
        Initialize the normalizer with a specific method and reference image path.
        If parameters are None, it tries to read them from environment variables.
        
        Args:
            method (str): Normalization method ('reinhard', 'histogram', 'clahe', 'hybrid').
            reference_path (str): Path to the reference standard image.
        """
        self.method = method or os.getenv("FACE_NORMALIZE_METHOD", "reinhard").lower()
        self.reference_path = reference_path or os.getenv("FACE_REF_IMAGE_PATH", "data/_face_ID/duong.jpg")
        
        # Reference image properties (will be computed in fit())
        self.ref_image_bgr: Optional[np.ndarray] = None
        self.ref_image_lab: Optional[np.ndarray] = None
        
        # Reinhard statistics
        self.ref_l_mean: float = 0.0
        self.ref_l_std: float = 1.0
        self.ref_a_mean: float = 0.0
        self.ref_a_std: float = 1.0
        self.ref_b_mean: float = 0.0
        self.ref_b_std: float = 1.0
        
        logger.info(f"ColorNormalizer initialized using method: '{self.method}'")
        
        # Auto-fit if reference path exists
        if self.reference_path and os.path.exists(self.reference_path):
            self.fit(self.reference_path)
        else:
            logger.warning(f"Reference image path '{self.reference_path}' does not exist yet. Please call fit() before transform().")

    def fit(self, reference_path: str) -> "FaceImageNormalizer":
        """
        Fits the normalizer by loading the reference image and pre-calculating statistics.
        This optimizes execution, ensuring we only compute reference parameters once.
        
        Args:
            reference_path (str): Path to the reference standard image.
            
        Returns:
            self: The normalizer instance.
        """
        try:
            logger.info(f"Fitting normalizer to reference image: {reference_path}")
            if not os.path.exists(reference_path):
                raise FileNotFoundError(f"Reference image file not found at {reference_path}")
                
            self.reference_path = reference_path
            # Read reference image in BGR
            self.ref_image_bgr = cv2.imread(reference_path)
            if self.ref_image_bgr is None:
                raise ValueError(f"Failed to read reference image from {reference_path}")
                
            # Convert reference image to LAB
            self.ref_image_lab = cv2.cvtColor(self.ref_image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Pre-compute Mean and Std Dev for Reinhard method
            ref_means, ref_stds = cv2.meanStdDev(self.ref_image_lab)
            
            # Extract channel-specific statistics
            self.ref_l_mean = float(ref_means[0][0])
            self.ref_l_std = float(ref_stds[0][0])
            self.ref_a_mean = float(ref_means[1][0])
            self.ref_a_std = float(ref_stds[1][0])
            self.ref_b_mean = float(ref_means[2][0])
            self.ref_b_std = float(ref_stds[2][0])
            
            # Prevent zero std deviation to avoid division by zero
            self.ref_l_std = max(self.ref_l_std, 1e-5)
            self.ref_a_std = max(self.ref_a_std, 1e-5)
            self.ref_b_std = max(self.ref_b_std, 1e-5)
            
            logger.info("Successfully fitted reference image statistics:")
            logger.info(f"  L_mean={self.ref_l_mean:.2f}, L_std={self.ref_l_std:.2f}")
            logger.info(f"  A_mean={self.ref_a_mean:.2f}, A_std={self.ref_a_std:.2f}")
            logger.info(f"  B_mean={self.ref_b_mean:.2f}, B_std={self.ref_b_std:.2f}")
            
            return self
        except Exception as e:
            logger.error(f"Error during fitting of reference image: {e}", exc_info=True)
            raise

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes the input image (in BGR format) based on the fitted reference statistics.
        
        Args:
            image (np.ndarray): Input BGR image.
            
        Returns:
            np.ndarray: Normalized BGR image.
        """
        if self.ref_image_bgr is None:
            raise RuntimeError("Normalizer is not fitted yet. Call fit() first.")
            
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image provided to transform()")

        try:
            if self.method == "reinhard":
                return self._reinhard_transfer(image)
            elif self.method == "histogram":
                return self._histogram_matching(image)
            elif self.method == "clahe":
                return self._clahe_only(image)
            elif self.method == "hybrid":
                return self._hybrid_transfer(image)
            else:
                logger.warning(f"Unknown method '{self.method}'. Returning original image.")
                return image
        except Exception as e:
            logger.error(f"Error occurred during image transformation: {e}", exc_info=True)
            # Return original image as fallback to prevent pipeline failure
            return image

    def transform_file(self, src_path: str, dest_path: str) -> bool:
        """
        Helper method to read an image file, transform it, and write the output.
        
        Args:
            src_path (str): Path to input image.
            dest_path (str): Path to save normalized image.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(src_path):
                logger.error(f"Source file not found at {src_path}")
                return False
                
            img = cv2.imread(src_path)
            if img is None:
                logger.error(f"Failed to read image from {src_path}")
                return False
                
            normalized_img = self.transform(img)
            
            # Make sure destination folder exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            success = cv2.imwrite(dest_path, normalized_img)
            if success:
                logger.debug(f"Saved normalized image to {dest_path}")
            else:
                logger.error(f"Failed to write image to {dest_path}")
            return success
        except Exception as e:
            logger.error(f"Failed to transform image file from {src_path} to {dest_path}: {e}")
            return False

    def _reinhard_transfer(self, src_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Reinhard Color Transfer in LAB space.
        """
        # Convert source to LAB and float32
        src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate mean and std of source channels
        src_means, src_stds = cv2.meanStdDev(src_lab)
        src_l_mean, src_l_std = float(src_means[0][0]), float(src_stds[0][0])
        src_a_mean, src_a_std = float(src_means[1][0]), float(src_stds[1][0])
        src_b_mean, src_b_std = float(src_means[2][0]), float(src_stds[2][0])
        
        # Prevent division by zero
        src_l_std = max(src_l_std, 1e-5)
        src_a_std = max(src_a_std, 1e-5)
        src_b_std = max(src_b_std, 1e-5)
        
        # Split channels
        l, a, b = cv2.split(src_lab)
        
        # Standardize and map to reference distribution
        l = ((l - src_l_mean) * (self.ref_l_std / src_l_std)) + self.ref_l_mean
        a = ((a - src_a_mean) * (self.ref_a_std / src_a_std)) + self.ref_a_mean
        b = ((b - src_b_mean) * (self.ref_b_std / src_b_std)) + self.ref_b_mean
        
        # Clip to valid range [0, 255]
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # Merge and convert back to BGR uint8
        transfer = cv2.merge([l, a, b]).astype(np.uint8)
        return cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)

    def _histogram_matching(self, src_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Histogram Matching using scikit-image on RGB channels.
        """
        # skimage match_histograms expects RGB or similar channel layout
        src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(self.ref_image_bgr, cv2.COLOR_BGR2RGB)
        
        # Match histograms
        matched_rgb = match_histograms(src_rgb, ref_rgb, channel_axis=-1)
        
        # Convert back to BGR uint8
        return cv2.cvtColor(matched_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _clahe_only(self, src_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel.
        """
        # Convert to LAB
        src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(src_lab)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        
        # Merge and convert back
        merged = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _hybrid_transfer(self, src_bgr: np.ndarray) -> np.ndarray:
        """
        Hybrid method: First apply Reinhard Color Transfer to match global color tone/distribution,
        then apply a mild CLAHE on the L channel to enhance face contrast and lighting details.
        """
        # Step 1: Apply Reinhard
        reinhard_img = self._reinhard_transfer(src_bgr)
        
        # Step 2: Apply mild CLAHE on top of it (using lower clipLimit for subtlety)
        return self._clahe_only(reinhard_img, clip_limit=1.5, tile_grid_size=(8, 8))
