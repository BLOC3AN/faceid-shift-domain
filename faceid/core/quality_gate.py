import cv2
import numpy as np
import logging
import threading
from typing import Dict, Any, Tuple, Optional
from insightface.app import FaceAnalysis

logger = logging.getLogger("FaceClustering.FaceQualityGate")

class FaceQualityGate:
    """
    Evaluates face image quality before FaceID verification.
    Calculates blur, brightness, contrast, pose, and size.
    """
    def __init__(self, model_root: str = ".", min_score: float = 0.4, good_score: float = 0.7):
        self.model_root = model_root
        self.min_score = min_score
        self.good_score = good_score
        self.app: Optional[FaceAnalysis] = None
        self._lock = threading.Lock()

    def _init_detector(self) -> None:
        """Lazy initialization of the face detector (only detection module to be fast)."""
        with self._lock:
            if self.app is None:
                try:
                    logger.info("Initializing detector for Quality Gate (detection module only)...")
                    self.app = FaceAnalysis(name='buffalo_m', root=self.model_root, allowed_modules=['detection'])
                    self.app.prepare(ctx_id=-1, det_size=(320, 320))  # Smaller detection size for speed
                    logger.info("Quality Gate detector initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize FaceAnalysis for Quality Gate: {e}")
                    raise

    def estimate_pose_from_kps(self, kps: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimates yaw, pitch, and roll in degrees from 5 face keypoints.
        kps structure: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        # 1. Roll (tilt): tilt of the line connecting both eyes
        left_eye = kps[0]
        right_eye = kps[1]
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        roll = float(np.arctan2(dy, dx) * 180.0 / np.pi)

        # 2. Yaw (turn left/right): horizontal ratio of nose to eyes
        nose = kps[2]
        d_left = float(abs(nose[0] - left_eye[0]))
        d_right = float(abs(nose[0] - right_eye[0]))
        yaw_ratio = (d_left - d_right) / (d_left + d_right + 1e-5)
        # Map to approx degrees (-90 to 90)
        yaw = float(yaw_ratio * 90.0)

        # 3. Pitch (tilt up/down): vertical ratio of nose between eyes and mouth
        eye_center = (left_eye + right_eye) / 2.0
        left_mouth = kps[3]
        right_mouth = kps[4]
        mouth_center = (left_mouth + right_mouth) / 2.0
        
        d_eye_nose = float(abs(nose[1] - eye_center[1]))
        d_nose_mouth = float(abs(mouth_center[1] - nose[1]))
        pitch_ratio = (d_eye_nose - d_nose_mouth) / (d_eye_nose + d_nose_mouth + 1e-5)
        # Map to approx degrees (-45 to 45)
        pitch = float(pitch_ratio * 45.0)

        return yaw, pitch, roll

    def evaluate(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates the quality of a raw BGR face image.
        Returns a dictionary containing sub-scores and the final consolidated quality_score.
        """
        if img is None or img.size == 0:
            return {
                "quality_score": 0.0,
                "is_valid": False,
                "reason": "Empty image",
                "metrics": {}
            }

        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Blur Score (Laplacian variance)
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Normalize: blur_val >= 120 is considered good (1.0), blur_val <= 15 is bad (0.0)
        s_blur = float(np.clip((blur_val - 15.0) / (120.0 - 15.0), 0.0, 1.0))

        # 2. Brightness Score (Ideal range 90-170)
        mean_brightness = float(np.mean(gray))
        # Distance from the center of ideal range (130)
        dist_brightness = abs(mean_brightness - 130.0)
        # Normalize: score decreases as it moves away from 130
        s_brightness = float(np.clip(1.0 - (dist_brightness / 100.0), 0.0, 1.0))

        # 3. Contrast Score (Ideal std dev >= 55)
        std_contrast = float(np.std(gray))
        s_contrast = float(np.clip(std_contrast / 55.0, 0.0, 1.0))

        # 4. Face Detection, Landmark validation & Pose estimation
        self._init_detector()
        faces = self.app.get(img)

        if not faces:
            return {
                "quality_score": 0.0,
                "is_valid": False,
                "reason": "No face detected in crop",
                "metrics": {
                    "blur": blur_val,
                    "brightness": mean_brightness,
                    "contrast": std_contrast,
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "bbox_size": 0.0
                }
            }

        # Pick largest face
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        bbox = face.bbox
        kps = face.kps

        # Check landmarks validity (must have 5 keypoints)
        if kps is None or len(kps) < 5:
            return {
                "quality_score": 0.0,
                "is_valid": False,
                "reason": "Invalid landmarks",
                "metrics": {
                    "blur": blur_val,
                    "brightness": mean_brightness,
                    "contrast": std_contrast,
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "bbox_size": 0.0
                }
            }

        # 5. Pose Score (ideal straight face yaw=0, pitch=0)
        yaw, pitch, roll = self.estimate_pose_from_kps(kps)
        # Penalize if yaw > 35 degrees or pitch > 25 degrees
        s_pose = float(np.clip(1.0 - (abs(yaw) / 35.0) - (abs(pitch) / 25.0), 0.0, 1.0))

        # 6. Bbox Size Score (Ideal width/height >= 112 pixels)
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        bbox_size = float(min(face_w, face_h))
        s_bbox = float(np.clip(bbox_size / 112.0, 0.0, 1.0))

        # Consolidated Quality Score (weighted average of components)
        # Weights: Blur (35%), Pose (25%), Size (15%), Contrast (15%), Brightness (10%)
        quality_score = float(
            0.35 * s_blur + 
            0.25 * s_pose + 
            0.15 * s_bbox + 
            0.15 * s_contrast + 
            0.10 * s_brightness
        )

        is_valid = quality_score >= self.min_score
        reason = "Pass" if is_valid else f"Low quality score ({quality_score:.2f} < {self.min_score})"

        return {
            "quality_score": quality_score,
            "is_valid": is_valid,
            "reason": reason,
            "metrics": {
                "blur": blur_val,
                "brightness": mean_brightness,
                "contrast": std_contrast,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "bbox_size": bbox_size
            }
        }
