"""
Main module for the facial expression recognition system.
Uses MediaPipe for face detection and DeepFace for expression recognition.
"""

import cv2
import numpy as np
import argparse
import os
import logging
from typing import Tuple, Optional, List, Dict
import mediapipe as mp
from deepface import DeepFace
import time
from collections import Counter
from collections import deque

import config
import utils
import statistics

# Set up logging
logger = utils.setup_logging()

class FaceDetector:
    """Face detector using MediaPipe Face Detection."""

    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize the MediaPipe face detector.

        Args:
            min_detection_confidence: Minimum confidence value for face detection
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full-range detection (5 meters)
            min_detection_confidence=min_detection_confidence
        )
        logger.info("MediaPipe Face Detection initialized")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.

        Args:
            frame: Frame to detect faces in

        Returns:
            List of face coordinates (x, y, w, h)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            height, width, _ = frame.shape
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box

                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                # Check if face size meets minimum requirements
                if w >= config.MIN_FACE_SIZE[0] and h >= config.MIN_FACE_SIZE[1]:
                    faces.append((x, y, w, h))

        return faces

class ExpressionRecognizer:
    """Expression recognizer using DeepFace."""

    def __init__(self, detector_backend="yolov8"):
        """
        Initialize the expression recognizer.

        Args:
            detector_backend: Backend detector to use for face detection
        """
        # Store the detector backend
        self.detector_backend = detector_backend
        logger.info(f"DeepFace expression recognizer initialized with {detector_backend} backend")

        # Map DeepFace emotions to our config emotions
        self.emotion_map = {
            "angry": "Angry",
            "disgust": "Disgust",
            "fear": "Fear",
            "happy": "Happy",
            "sad": "Sad",
            "surprise": "Surprise",
            "neutral": "Neutral"
        }

        # Warm up DeepFace to avoid first-run delay
        try:
            # Create a small blank image for warmup
            warmup_img = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = DeepFace.analyze(
                warmup_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend=self.detector_backend
            )
            logger.info(f"DeepFace warmed up successfully with {detector_backend} backend")
        except Exception as e:
            logger.warning(f"DeepFace warmup failed with {detector_backend} backend: {e}")

    def predict_expression(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Predict the expression from a face image.

        Args:
            face_img: Face image

        Returns:
            Tuple of predicted expression and confidence
        """
        try:
            # Skip if face is too small or empty
            if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                logger.warning("Empty face region")
                return "Unknown", 0.0

            # Analyze face with DeepFace
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend=self.detector_backend
            )

            # Get emotion data
            emotion_data = result[0]['emotion']

            # Sort emotions by confidence to find top two
            sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)

            # Get top emotion
            top_emotion_name, top_confidence = sorted_emotions[0]

            # If top emotion is "neutral", use the second highest confidence emotion instead
            if top_emotion_name == "neutral" and len(sorted_emotions) > 1:
                top_emotion_name, top_confidence = sorted_emotions[1]

            # Map to our emotion names
            mapped_emotion = self.emotion_map.get(top_emotion_name, top_emotion_name)

            return mapped_emotion, top_confidence / 100.0  # Convert percentage to 0-1 scale

        except Exception as e:
            logger.error(f"Error during prediction with {self.detector_backend}: {e}")
            return "Unknown", 0.0

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Facial Expression Recognition System")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file. If not provided, webcam will be used"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_ID,
        help=f"Camera device ID (default: {config.CAMERA_ID})"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output video file"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=True,
        help="Display video feed (default: True)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=False,
        help="Generate statistics and charts after processing"
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        help="Directory to save statistics charts"
    )
    parser.add_argument(
        "--detector-backend",
        type=str,
        choices=['yolov8', 'yolov11n', 'yolov11s', 'yolov11m', 'all'],
        default="all",
        help="Detector backend to use (default: all)"
    )

    return parser.parse_args()

def setup_video_capture(args) -> Tuple[cv2.VideoCapture, Optional[cv2.VideoWriter]]:
    """
    Set up video capture and writer.

    Args:
        args: Command line arguments

    Returns:
        Tuple[cv2.VideoCapture, Optional[cv2.VideoWriter]]: Video capture and writer objects
    """

    # Set up video capture
    if args.video:
        logger.info(f"Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        logger.info(f"Opening camera device: {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH // 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT // 2)
        cap.set(cv2.CAP_PROP_FPS, config.FPS)

    # Check if video opened successfully
    if not cap.isOpened():
        logger.error("Error: Could not open video source")
        raise ValueError("Could not open video source")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Video properties: {width}x{height} @ {fps}fps")

    # Set up video writer if output file specified
    writer = None
    if args.output:
        logger.info(f"Setting up video writer: {args.output}")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    return cap, writer

def process_frame(frame, face_detector, expression_recognizer, frame_number):
    """
    Process a single frame: detect faces and recognize expressions.

    Args:
        frame: Frame to process
        face_detector: Face detector object
        expression_recognizer: Expression recognizer object
        frame_number: Current frame number

    Returns:
        Processed frame with annotations
    """
    global expression_queue
    global all_expressions_data

    # Detect faces
    faces = face_detector.detect_faces(frame)

    # Process each face
    for face_coords in faces:
        x, y, w, h = face_coords

        # Extract face region
        face_img = frame[y:y+h, x:x+w]

        # Skip if face is too small
        if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
            continue

        # Predict expression from the face image
        expression, confidence = expression_recognizer.predict_expression(face_img)

        # If the confidence is above the threshold, append the tuple to the deque
        if confidence >= config.FACE_CONFIDENCE_THRESHOLD:
            expression_queue.append((expression, confidence))
            # Also store in the all_expressions_data list for statistics
            all_expressions_data.append((frame_number, expression, confidence))

        # Extract all expressions from the deque (now iterable)
        expressions = [exp for exp, conf in expression_queue]

        # Determine the most common expression using Counter
        if expressions:
            most_common_expression, _ = Counter(expressions).most_common(1)[0]
            # Calculate average confidence for the most common expression
            confidences = [conf for exp, conf in expression_queue if exp == most_common_expression]
            mean_confidence = sum(confidences) / len(confidences)
        else:
            most_common_expression = "Unknown"  # Default expression
            mean_confidence = 0.0

        mean_expression = most_common_expression

        # Draw face box with emotion
        frame = utils.draw_face_box_with_expression(frame, face_coords, mean_expression, mean_confidence)

    return frame

def run_detector_backend(args, detector_backend):
    """
    Run the facial expression recognition system with a specific detector backend.

    Args:
        args: Command line arguments
        detector_backend: Detector backend to use

    Returns:
        List of expression data tuples
    """
    global expression_queue
    global all_expressions_data
    
    logger.info(f"Starting facial expression recognition with {detector_backend} backend")
    
    # Initialize data structures
    expression_queue = deque(maxlen=50)
    all_expressions_data = []
    
    try:
        # Load face detector
        logger.info("Loading MediaPipe face detector")
        face_detector = FaceDetector(min_detection_confidence=config.FACE_CONFIDENCE_THRESHOLD)

        # Load expression recognizer with specified backend
        logger.info(f"Loading DeepFace expression recognizer with {detector_backend} backend")
        expression_recognizer = ExpressionRecognizer(detector_backend=detector_backend)

        # Set up video capture and writer
        cap, writer = setup_video_capture(args)

        # Create FPS counter
        fps_counter = utils.FPSCounter()

        logger.info(f"Starting video processing loop with {detector_backend} backend")

        # Initialize frame counter
        frame_number = 0

        # Main loop
        while True:
            # Read frame
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

            # Break if end of video
            if not ret:
                logger.info("End of video")
                break

            # Process frame
            processed_frame = process_frame(frame, face_detector, expression_recognizer, frame_number)

            # Update and draw FPS
            fps = fps_counter.update()
            processed_frame = fps_counter.draw_fps(processed_frame)

            # Write frame to output file if specified
            if writer:
                writer.write(processed_frame)

            # Display frame if specified
            if args.display:
                window_name = f'Facial Expression Recognition - {detector_backend}'
                cv2.imshow(window_name, processed_frame)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User interrupted")
                    break

            # Increment frame counter
            frame_number += 1

        logger.info(f"Video processing completed with {detector_backend} backend")
        
        # Return the collected expression data
        return all_expressions_data
    
    except Exception as e:
        logger.error(f"Error in {detector_backend} backend: {e}", exc_info=True)
        return []
    
    finally:
        # Clean up
        logger.info(f"Cleaning up resources for {detector_backend} backend")
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'writer' in locals() and writer is not None:
            writer.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the facial expression recognition system."""
    # Parse command line arguments
    args = parse_arguments()

    logger.info("Starting facial expression recognition system")

    # Determine which detector backends to use
    detector_backends = ['yolov8'] if args.detector_backend == 'all' else [args.detector_backend]
    
    # Dictionary to store expression data for each backend
    all_backend_data = {}
    
    # Run each detector backend
    for backend in detector_backends:
        # Run the detector backend
        expression_data = run_detector_backend(args, backend)
        
        # Store the expression data
        all_backend_data[backend] = expression_data
        
        # Generate statistics if requested
        if args.stats and expression_data:
            logger.info(f"Generating statistics and charts for {backend} backend")
            stats_dir = args.stats_dir or os.path.join(config.BASE_DIR, "statistics")
            charts = statistics.analyze_video_expressions(expression_data, stats_dir, backend)
            logger.info(f"Statistics generated for {backend} backend. Charts saved to {os.path.join(stats_dir, backend)}")
            
            # Print chart paths
            for chart_type, path in charts.items():
                if isinstance(path, list):
                    logger.info(f"{chart_type}: {len(path)} charts generated")
                elif path:
                    logger.info(f"{chart_type}: {path}")
    
    logger.info("All detector backends completed")

if __name__ == "__main__":
    main()

