import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

class ExerciseValidator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Exercise configurations
        self.exercise_configs = {
            'arm_raise': {
                'angle_threshold': 160,  # degrees
                'min_angle': 60,
                'max_angle': 180,
                'landmarks': ['shoulder', 'elbow', 'wrist'],
                'rep_threshold': 0.8  # confidence threshold for rep counting
            },
            'squat': {
                'angle_threshold': 120,  # degrees
                'min_angle': 60,
                'max_angle': 180,
                'landmarks': ['hip', 'knee', 'ankle'],
                'rep_threshold': 0.8
            },
            'jumping_jack': {
                'arm_angle_threshold': 150,  # arm angle
                'leg_angle_threshold': 160,  # leg angle
                'min_arm_angle': 60,
                'max_arm_angle': 180,
                'min_leg_angle': 120,
                'max_leg_angle': 180,
                'rep_threshold': 0.8
            }
        }
    
    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate the angle between three points."""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
                
            return angle
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 0.0
    
    def get_landmark_coordinates(self, results, landmark_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a specific landmark."""
        landmark_mapping = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        }
        
        if landmark_name not in landmark_mapping:
            return None
            
        landmark = results.pose_landmarks.landmark[landmark_mapping[landmark_name]]
        return (landmark.x, landmark.y)
    
    def detect_arm_raise(self, results) -> Dict:
        """Detect and count arm raise repetitions."""
        left_shoulder = self.get_landmark_coordinates(results, 'left_shoulder')
        left_elbow = self.get_landmark_coordinates(results, 'left_elbow')
        left_wrist = self.get_landmark_coordinates(results, 'left_wrist')
        
        right_shoulder = self.get_landmark_coordinates(results, 'right_shoulder')
        right_elbow = self.get_landmark_coordinates(results, 'right_elbow')
        right_wrist = self.get_landmark_coordinates(results, 'right_wrist')
        
        if not all([left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist]):
            return {'reps': 0, 'confidence': 0.0, 'current_angle': 0.0, 'valid_movement': False}
        
        # Calculate angles for both arms
        left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Use the average angle
        avg_angle = (left_angle + right_angle) / 2
        
        config = self.exercise_configs['arm_raise']
        confidence = 1.0 if config['min_angle'] <= avg_angle <= config['max_angle'] else 0.5
        
        # Valid movement: arms raised (avg_angle >= min_angle)
        valid_movement = avg_angle >= config['min_angle']
        
        return {
            'reps': 0,  # Will be calculated in the main processing loop
            'confidence': confidence,
            'current_angle': avg_angle,
            'is_rep_position': avg_angle >= config['angle_threshold'],
            'valid_movement': valid_movement
        }
    
    def detect_squat(self, results) -> Dict:
        """Detect and count squat repetitions."""
        left_hip = self.get_landmark_coordinates(results, 'left_hip')
        left_knee = self.get_landmark_coordinates(results, 'left_knee')
        left_ankle = self.get_landmark_coordinates(results, 'left_ankle')
        
        right_hip = self.get_landmark_coordinates(results, 'right_hip')
        right_knee = self.get_landmark_coordinates(results, 'right_knee')
        right_ankle = self.get_landmark_coordinates(results, 'right_ankle')
        
        if not all([left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
            return {'reps': 0, 'confidence': 0.0, 'current_angle': 0.0, 'valid_movement': False}
        
        # Calculate angles for both legs
        left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Use the average angle
        avg_angle = (left_angle + right_angle) / 2
        
        config = self.exercise_configs['squat']
        confidence = 1.0 if config['min_angle'] <= avg_angle <= config['max_angle'] else 0.5
        
        # Valid movement: legs bent (avg_angle <= 160)
        valid_movement = avg_angle <= 160
        
        return {
            'reps': 0,
            'confidence': confidence,
            'current_angle': avg_angle,
            'is_rep_position': avg_angle <= config['angle_threshold'],
            'valid_movement': valid_movement
        }
    
    def detect_jumping_jack(self, results) -> Dict:
        """Detect and count jumping jack repetitions."""
        # Arm angles
        left_shoulder = self.get_landmark_coordinates(results, 'left_shoulder')
        left_elbow = self.get_landmark_coordinates(results, 'left_elbow')
        left_wrist = self.get_landmark_coordinates(results, 'left_wrist')
        
        right_shoulder = self.get_landmark_coordinates(results, 'right_shoulder')
        right_elbow = self.get_landmark_coordinates(results, 'right_elbow')
        right_wrist = self.get_landmark_coordinates(results, 'right_wrist')
        
        # Leg angles
        left_hip = self.get_landmark_coordinates(results, 'left_hip')
        left_knee = self.get_landmark_coordinates(results, 'left_knee')
        left_ankle = self.get_landmark_coordinates(results, 'left_ankle')
        
        right_hip = self.get_landmark_coordinates(results, 'right_hip')
        right_knee = self.get_landmark_coordinates(results, 'right_knee')
        right_ankle = self.get_landmark_coordinates(results, 'right_ankle')
        
        if not all([left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist,
                   left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
            return {'reps': 0, 'confidence': 0.0, 'arm_angle': 0.0, 'leg_angle': 0.0, 'valid_movement': False}
        
        # Calculate angles
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        
        left_leg_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        avg_leg_angle = (left_leg_angle + right_leg_angle) / 2
        
        config = self.exercise_configs['jumping_jack']
        
        # Check if both arms and legs are in the correct position
        arms_up = avg_arm_angle >= config['arm_angle_threshold']
        legs_spread = avg_leg_angle >= config['leg_angle_threshold']
        
        confidence = 1.0 if (arms_up and legs_spread) else 0.5
        
        # Valid movement: arms raised OR legs spread (avg_arm_angle >= min_arm_angle OR avg_leg_angle >= min_leg_angle)
        valid_movement = (avg_arm_angle >= config['min_arm_angle']) or (avg_leg_angle >= config['min_leg_angle'])
        
        return {
            'reps': 0,
            'confidence': confidence,
            'arm_angle': avg_arm_angle,
            'leg_angle': avg_leg_angle,
            'is_rep_position': arms_up and legs_spread,
            'valid_movement': valid_movement
        }
    
    def count_repetitions(self, exercise_type: str, detection_results: List[Dict]) -> Dict:
        """Count repetitions based on detection results."""
        if not detection_results:
            return {'total_reps': 0, 'sets': 0, 'accuracy': 0.0}
        
        config = self.exercise_configs.get(exercise_type, {})
        rep_threshold = config.get('rep_threshold', 0.8)
        
        total_reps = 0
        current_set = 0
        in_rep_position = False
        confidence_scores = []
        
        for result in detection_results:
            confidence_scores.append(result.get('confidence', 0.0))
            
            if result.get('is_rep_position', False) and not in_rep_position:
                if result.get('confidence', 0.0) >= rep_threshold:
                    total_reps += 1
                    in_rep_position = True
            elif not result.get('is_rep_position', False):
                in_rep_position = False
        
        # Calculate sets (assuming 10 reps per set for now)
        sets = (total_reps + 9) // 10  # Ceiling division
        
        # Calculate accuracy based on average confidence
        accuracy = np.mean(confidence_scores) * 100 if confidence_scores else 0.0
        
        return {
            'total_reps': total_reps,
            'sets': sets,
            'accuracy': round(accuracy, 2)
        }
    
    def process_video(self, video_path: str, exercise_type: str) -> Dict:
        """Process video and return exercise validation results."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}
            
            detection_results = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 3rd frame to speed up processing
                if frame_count % 3 != 0:
                    frame_count += 1
                    continue
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    if exercise_type == 'arm_raise':
                        result = self.detect_arm_raise(results)
                    elif exercise_type == 'squat':
                        result = self.detect_squat(results)
                    elif exercise_type == 'jumping_jack':
                        result = self.detect_jumping_jack(results)
                    else:
                        result = {'reps': 0, 'confidence': 0.0, 'valid_movement': False}
                    
                    detection_results.append(result)
                
                frame_count += 1
            
            cap.release()
            
            # Check if video contains valid movements for the assigned exercise
            valid_movements = [res.get("valid_movement", False) for res in detection_results]
            
            if not any(valid_movements):
                return {
                    "exercise_type": exercise_type,
                    "success": False,
                    "error": "Exercise mismatch: The uploaded video does not match the assigned exercise.",
                    "frames_processed": len(detection_results),
                    "total_reps": 0,
                    "sets": 0,
                    "accuracy": 0.0
                }
            
            # Count repetitions
            rep_results = self.count_repetitions(exercise_type, detection_results)
            
            return {
                'exercise_type': exercise_type,
                'total_reps': rep_results['total_reps'],
                'sets': rep_results['sets'],
                'accuracy': rep_results['accuracy'],
                'frames_processed': len(detection_results),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                'error': str(e),
                'success': False
            }
        finally:
            self.pose.close()
    
    def get_exercise_type_from_name(self, exercise_name: str) -> str:
        """Map exercise names to internal exercise types."""
        exercise_name_lower = exercise_name.lower()
        
        if any(keyword in exercise_name_lower for keyword in ['arm', 'raise', 'lift']):
            return 'arm_raise'
        elif any(keyword in exercise_name_lower for keyword in ['squat', 'sit']):
            return 'squat'
        elif any(keyword in exercise_name_lower for keyword in ['jumping', 'jack']):
            return 'jumping_jack'
        else:
            # Default to arm raise for unknown exercises
            return 'arm_raise'
