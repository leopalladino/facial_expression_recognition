"""
Configuration settings for the facial expression recognition system.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.absolute()
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)

# Video settings
CAMERA_ID = 0  # Default camera (usually webcam)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Face detection settings
FACE_CONFIDENCE_THRESHOLD = 0.5
MIN_FACE_SIZE = (50, 50)  # Minimum face size to detect

# Expression recognition settings
# Removed "Neutral" from the list as requested
EXPRESSIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

# Display settings
FONT = "FONT_HERSHEY_SIMPLEX"
FONT_SCALE = 0.8
FONT_THICKNESS = 2
BOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White
BOX_COLOR = {
    "Angry": (0, 0, 255),      # Red
    "Disgust": (0, 140, 255),  # Orange
    "Fear": (0, 255, 255),     # Yellow
    "Happy": (0, 255, 0),      # Green
    "Sad": (255, 0, 0),        # Blue
    "Surprise": (255, 0, 255), # Magenta
    "Neutral": (255, 255, 255) # White
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOG_DIR, "facial_expression.log")

