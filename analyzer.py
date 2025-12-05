import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

class BiometricAnalyzer:
    def __init__(self):
        # Initialize Mediapipe Face Mesh
        print("Loading Mediapipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("Models loaded.")

    def get_landmarks(self, image):
        # Convert the BGR image to RGB before processing.
        # Check if already RGB (Streamlit PIL -> numpy is RGB usually, but cv2 is BGR)
        # We assume input is RGB or BGR, but Mediapipe needs RGB.
        # Let's standardize: assume input is numpy array.
        
        # NOTE: If we are unsure of color space, we should be careful.
        # App.py loads with PIL -> np.array (RGB).
        
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return None, None
            
        # Take the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array of (x, y) coordinates
        h, w, _ = image.shape
        landmarks = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append([x, y])
            
        return np.array(landmarks), None # No rect needed for Mediapipe usually, but keeping signature similar

    def calculate_ratios(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            return {}

        # Mediapipe Face Mesh Indices (468 points)
        # Key landmarks for ratios:
        
        # Left Eye Inner: 133, Outer: 33
        # Right Eye Inner: 362, Outer: 263
        # Nose Top (Between eyes/root): 168 (or 6 for verified center)
        # Nose Bottom (Tip/Base): 1 or 2. 2 is often used for tip.
        # Mouth Left: 61, Right: 291
        
        # Calculate distances
        
        # Inner Eye Distance (Inter-canthal): 133 to 362
        try:
            p_left_inner = landmarks[133]
            p_right_inner = landmarks[362]
            inner_eye_distance = np.linalg.norm(p_left_inner - p_right_inner)
            
            # Nose Length: Root (168) to Tip Infradentale? No, usually Glabella to Subnasale.
            # 168 is mid-way between eyes. 2 is tip.
            p_nose_root = landmarks[168]
            p_nose_tip = landmarks[2] # or 94 for tip consistency? 1 is standard tip. 2 is slightly lower.
            # Let's use 6 (mid eyes) to 1 (tip) for length.
            # Actually user logic was: 27 to 33 in Dlib. 27(glabella) to 33(subnasale).
            # Mediapipe eq: 10 (glabella/top) to 2 (nasal bounding)?
            # Let's use 168 (glabella approx) to 1 (tip of nose).
            nose_length = np.linalg.norm(landmarks[168] - landmarks[1])
            
            # Mouth Width: 61 to 291
            p_mouth_left = landmarks[61]
            p_mouth_right = landmarks[291]
            mouth_width = np.linalg.norm(p_mouth_left - p_mouth_right)

            ratios = {
                "eye_distance_nose_length": inner_eye_distance / nose_length if nose_length != 0 else 0,
                "nose_length_mouth_width": nose_length / mouth_width if mouth_width != 0 else 0
            }
            return ratios
        except IndexError:
            return {}

    def calculate_beauty_score(self, ratios):
        # Golden Ratio Approximation
        golden_ratio = 1.618
        score_penalty = 0
        
        for key, ratio in ratios.items():
            score_penalty += abs(golden_ratio - ratio)
            
        score = max(0, 10 - (score_penalty * 2.5)) 
        return score

    def evaluate_beauty_text(self, beauty_score):
        if beauty_score >= 9.0: return "Outstanding"
        elif beauty_score >= 8.0: return "Excellent"
        elif beauty_score >= 7.0: return "Good"
        elif beauty_score >= 5.0: return "Average"
        else: return "Below Average"

    def analyze_deepface(self, image):
        try:
            # DeepFace expects BGR or RGB. 
            # We standardize on converting to numpy array if not already.
            img_copy = image.copy()
            
            # DeepFace.analyze
            result = DeepFace.analyze(img_copy, actions=['age', 'gender'], enforce_detection=False, detector_backend='opencv') # faster backend
            if isinstance(result, list):
                result = result[0]
            
            # Formatting Gender: if dict (probabilities), extract max
            if isinstance(result.get('gender'), dict):
                gender_probs = result['gender']
                dominant_gender = max(gender_probs, key=gender_probs.get)
                result['gender'] = dominant_gender
                
            return result
        except Exception as e:
            print(f"DeepFace error: {e}")
            return {"age": 0, "gender": "Unknown"}

    def visualize(self, image, landmarks):
        if landmarks is None:
            return image
            
        viz_image = image.copy()
        
        # Draw specific feature points only (cleaner than all 468 dots)
        # We can draw the full mesh or just key points. Let's draw contours for a "tech" look.
        
        # Mediapipe connections for contours
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create a blank landmark object for reuse? 
        # Easier to just draw manually with cv2 for the specific calculated lines.
        
        # 1. Draw connecting lines used in calc
        # Eye inner 133-362
        cv2.line(viz_image, tuple(landmarks[133]), tuple(landmarks[362]), (255, 255, 0), 2) # Cyan
        
        # Nose 168-1
        cv2.line(viz_image, tuple(landmarks[168]), tuple(landmarks[1]), (255, 0, 255), 2) # Magenta
        
        # Mouth 61-291
        cv2.line(viz_image, tuple(landmarks[61]), tuple(landmarks[291]), (0, 255, 255), 2) # Yellow
        
        # 2. Draw subtle dots for key areas
        key_indices = [33, 133, 362, 263, 1, 61, 291, 168]
        for idx in key_indices:
             cv2.circle(viz_image, tuple(landmarks[idx]), 3, (0, 255, 0), -1)

        return viz_image

