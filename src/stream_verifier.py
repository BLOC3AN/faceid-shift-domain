import os
import sys
import cv2
import time
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

# Add project root and src to python path
project_root = "/home/hailt/Desktop/faceID_normalize"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from quality_gate import FaceQualityGate
from face_id_verifier import FaceIDVerifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceClustering.StreamVerifier")

def main():
    load_dotenv(os.path.join(project_root, ".env"))

    # Configs
    rtsp_url = os.getenv("RTSP_URL", "rtsp://admin:admin_123@192.168.1.103:1904/stream2")
    ref_image_path = os.path.join(project_root, os.getenv("FACE_REF_IMAGE_PATH", "data/_face_ID/duong_cropped.jpg"))
    model_root = project_root
    
    quality_min_score = float(os.getenv("QUALITY_MIN_SCORE", "0.4"))
    quality_good_score = float(os.getenv("QUALITY_GOOD_SCORE", "0.7"))
    
    # CCTV Live Stream optimized verification thresholds (adjusted due to Domain Shift without translation)
    verify_threshold_high = 0.45
    verify_threshold_standard = 0.38
    verify_threshold_default = 0.42

    output_match_dir = os.path.join(project_root, "data/stream_matches")
    os.makedirs(output_match_dir, exist_ok=True)

    logger.info("Initializing verifier and loading reference embedding...")
    # Initialize Verifier (which loads reference image and manages cache)
    verifier = FaceIDVerifier(
        ref_image_path=ref_image_path,
        model_root=model_root,
        verify_threshold=verify_threshold_default,
        verify_threshold_high=verify_threshold_high,
        verify_threshold_standard=verify_threshold_standard,
        quality_min_score=quality_min_score,
        quality_good_score=quality_good_score,
        calibration_enabled=False  # Disable domain calibration for single-frame live verification
    )
    
    ref_emb = verifier.ref_embedding
    if ref_emb is None:
        logger.error("Could not load reference embedding. Exiting.")
        return
        
    logger.info("Reference embedding loaded successfully.")

    # Initialize FaceQualityGate
    logger.info("Initializing Quality Gate...")
    quality_gate = FaceQualityGate(
        model_root=model_root,
        min_score=quality_min_score,
        good_score=quality_good_score
    )

    # Initialize InsightFace detector + recognizer for live stream
    logger.info("Initializing InsightFace Model for live stream...")
    app = FaceAnalysis(name='buffalo_m', root=model_root, allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    logger.info("Models initialized successfully.")

    logger.info(f"Opening RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        logger.error(f"Cannot open RTSP stream: {rtsp_url}")
        return

    logger.info("RTSP Stream opened. Starting detection loop. Press 'q' to quit.")
    
    frame_count = 0
    process_every_n_frames = 2  # Skip frames to keep up with realtime
    
    headless = False
    logger.info("RTSP Stream opened. Starting detection loop. Press 'q' to quit (or Ctrl+C in headless mode).")
    
    frame_count = 0
    process_every_n_frames = 2  # Skip frames to keep up with realtime
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame. Reconnecting in 2 seconds...")
            time.sleep(2)
            cap.open(rtsp_url)
            continue
            
        frame_count += 1
        
        # Decide processing frame
        should_process = frame_count % process_every_n_frames == 0
        
        if not should_process:
            if not headless:
                try:
                    cv2.imshow("RTSP FaceID Verifier - Live", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    logger.warning(f"Failed to display frame (switching to headless mode): {e}")
                    headless = True
            continue

        h, w, _ = frame.shape
        display_frame = frame.copy() if not headless else None
        
        # Detect faces
        faces = app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            # Clip bounding boxes
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
                
            # Crop face image for quality evaluation
            face_crop = frame[y1:y2, x1:x2]
            
            # 1. Quality Gate evaluation
            q_info = quality_gate.evaluate(face_crop)
            q_score = q_info["quality_score"]
            is_valid = q_info["is_valid"]
            reason = q_info["reason"]
            
            if not is_valid:
                logger.warning(f"Face rejected by Quality Gate: Q={q_score:.2f}. Reason: {reason}")
                if not headless and display_frame is not None:
                    # REJECTED (Low Quality) - draw Orange Box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        display_frame, 
                        f"Rejected: Q={q_score:.2f}", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 165, 255), 
                        1
                    )
                continue

            # 2. Extract embedding and L2-normalize
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)
            
            # 3. Compute similarity with reference
            similarity = float(np.dot(emb, ref_emb))
            
            # 4. Apply Adaptive Threshold
            threshold = verify_threshold_standard if q_score >= quality_good_score else verify_threshold_high
            is_match = similarity >= threshold
            
            if is_match:
                # MATCHED (Duong) - draw Green Box
                color = (0, 255, 0)
                name_tag = f"Duong (Sim: {similarity:.2f}, Q: {q_score:.2f})"
                logger.info(f"[MATCH] Duong detected! Sim: {similarity:.4f} >= Thresh: {threshold} (Q: {q_score:.2f})")
                
                # Save matching frame and cropped face for record
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                match_filename = f"duong_{timestamp}.jpg"
                cv2.imwrite(os.path.join(output_match_dir, match_filename), frame)
                
                cropped_dir = os.path.join(output_match_dir, "cropped")
                os.makedirs(cropped_dir, exist_ok=True)
                crop_filename = f"duong_{timestamp}_crop.jpg"
                cv2.imwrite(os.path.join(cropped_dir, crop_filename), face_crop)
                logger.info(f"Saved matching frame and cropped face: {match_filename} & {crop_filename}")
            else:
                # UNKNOWN - draw Red Box
                color = (0, 0, 255)
                name_tag = f"Unknown (Sim: {similarity:.2f}, Q: {q_score:.2f})"
                logger.info(f"[UNKNOWN] Face detected. Similarity: {similarity:.4f} < Thresh: {threshold} (Q: {q_score:.2f})")
                
            if not headless and display_frame is not None:
                # Draw bounding box and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame, 
                    name_tag, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
            
        # Display the result
        if not headless and display_frame is not None:
            try:
                cv2.imshow("RTSP FaceID Verifier - Live", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                logger.warning(f"Failed to display frame (switching to headless mode): {e}")
                headless = True
            
    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    logger.info("Stream processing stopped.")

if __name__ == "__main__":
    main()
