import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app import FaceAnalysis

def main():
    src_path = "data/_face_ID/duong.jpg"
    dest_path = "data/_face_ID/duong_cropped.jpg"
    
    if not os.path.exists(src_path):
        print(f"Error: {src_path} not found")
        return
        
    print(f"Loading reference image: {src_path}")
    img = cv2.imread(src_path)
    if img is None:
        print("Error: Failed to read image")
        return
        
    print("Initializing FaceAnalysis...")
    app = FaceAnalysis(name='buffalo_m', root='.')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    print("Detecting face in reference image...")
    faces = app.get(img)
    if not faces:
        print("Error: No face detected in reference image")
        return
        
    # Sort by size and pick the largest
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    face = faces[0]
    
    # Get bounding box coordinates
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Add a larger padding (e.g. 40%) to give the detector enough context (head/shoulders)
    h, w = img.shape[:2]
    padding_w = int((x2 - x1) * 0.4)
    padding_h = int((y2 - y1) * 0.4)
    
    x1_pad = max(0, x1 - padding_w)
    y1_pad = max(0, y1 - padding_h)
    x2_pad = min(w, x2 + padding_w)
    y2_pad = min(h, y2 + padding_h)
    
    print(f"Cropping face with padding: [{x1_pad}, {y1_pad}, {x2_pad}, {y2_pad}]")
    cropped_img = img[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Save cropped image
    cv2.imwrite(dest_path, cropped_img)
    print(f"Successfully saved cropped reference face to: {dest_path}")

if __name__ == "__main__":
    main()
