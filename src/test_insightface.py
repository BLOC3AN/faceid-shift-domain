import os
import cv2
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app import FaceAnalysis

def test():
    img_path = "data/_face_ID/duong.jpg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return
        
    print(f"Loading image {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Failed to read image")
        return
        
    print("Initializing FaceAnalysis with buffalo_m...")
    # Initialize using local models folder
    app = FaceAnalysis(name='buffalo_m', root='.')
    app.prepare(ctx_id=-1, det_size=(640, 640)) # ctx_id=-1 for CPU
    
    print("Detecting and extracting embedding...")
    faces = app.get(img)
    print(f"Found {len(faces)} faces.")
    for idx, face in enumerate(faces):
        print(f"Face {idx+1}:")
        print(f"  Bounding Box: {face.bbox}")
        print(f"  Embedding Shape: {face.embedding.shape if face.embedding is not None else 'None'}")
        if face.embedding is not None:
            # Print first 5 dimensions of embedding
            print(f"  First 5 dims: {face.embedding[:5]}")

if __name__ == "__main__":
    test()
