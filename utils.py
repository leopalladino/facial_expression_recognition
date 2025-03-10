"""
Utility functions for the facial expression recognition system.
"""

import cv2
import logging
import time
from typing import Tuple, List
import config
import numpy as np

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("facial_expression")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # Create file handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def draw_face_box_with_expression(
    frame: np.ndarray,
    face_coords: Tuple[int, int, int, int],
    expression: str,
    confidence: float
) -> np.ndarray:
    """
    Draw a bounding box around a face and display the predicted expression.

    Args:
        frame: The video frame to draw on
        face_coords: (x, y, w, h) coordinates of the face
        expression: Predicted expression label
        confidence: Confidence score of the prediction

    Returns:
        Frame with bounding box and expression label
    """
    x, y, w, h = face_coords

    # Get color based on expression
    color = config.BOX_COLOR.get(expression, (255, 255, 255))

    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.BOX_THICKNESS)

    # Prepare text with expression and confidence
    text = f"{expression} ({confidence:.2f})"

    # Get text size to position it properly
    font = getattr(cv2, config.FONT)
    text_size = cv2.getTextSize(text, font, config.FONT_SCALE, config.FONT_THICKNESS)[0]

    # Draw filled rectangle for text background
    cv2.rectangle(
        frame,
        (x, y - text_size[1] - 10),
        (x + text_size[0], y),
        color,
        -1  # Filled rectangle
    )

    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y - 5),
        font,
        config.FONT_SCALE,
        config.TEXT_COLOR,
        config.FONT_THICKNESS
    )

    return frame

class FPSCounter:
    """Class to calculate and display FPS."""

    def __init__(self):
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def update(self) -> float:
        """
        Update FPS calculation.

        Returns:
            float: Current FPS
        """
        current_time = time.time()
        self.frame_count += 1

        # Update FPS every second
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.prev_time)
            self.prev_time = current_time
            self.frame_count = 0

        return self.fps

    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw FPS on the frame.

        Args:
            frame: Frame to draw on

        Returns:
            Frame with FPS text
        """
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            getattr(cv2, config.FONT),
            config.FONT_SCALE,
            (0, 255, 0),  # Green color for FPS
            config.FONT_THICKNESS
        )
        return frame

