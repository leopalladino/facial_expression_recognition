### Facial Expression Recognition with YOLOv8

This project analyzes video footage to identify instances of facial expressions using YOLOv8. The system processes video frames in real-time, detecting faces and generating comprehensive statistics about detection accuracy and performance.

## Example Scenarios

### Scenario 1: Reaction to Provocative Questions

![Demo](statistics/demo_angry/messi_angry.gif)

When analyzing footage of interview subjects facing challenging questions, our system can:

- Track facial movements and micro-expressions
- Identify objects in the frame (microphones, recording equipment)
- Measure subject positioning and movement patterns

![Demo](statistics/demo_angry/messi_angry_stats.jpeg)

*Figure 1: Facial expression recognition statistics during provocative questioning*

### Scenario 2: Positive Interaction Analysis

![Demo](statistics/demo_happy/messi_happy.gif)

For positive interactions, our system detects:

- Subject's gestures and body language
- Surrounding objects and their relation to the subject
- Environmental factors affecting detection accuracy

![Demo](statistics/demo_happy/messi_happy_stats.jpeg)

*Figure 2: Facial expression recognitions statistics during positive questioning*

## How It Works

The project uses YOLOv8 (You Only Look Once) for real-time object detection, offering:

- Fast processing speeds suitable for video analysis
- High accuracy in identifying multiple object classes
- Robust performance across varying lighting conditions

## Running the Project

You can run the project with different options to customize the analysis:

```shellscript
# Run with all detector backends and generate statistics
python main.py --video your_video.mp4 --stats

# Specify a custom directory for statistics
python main.py --video your_video.mp4 --stats --stats-dir ./my_stats

# Process a live camera feed instead of a video file
python main.py --camera 0 --stats
```

## Statistics Generation

The system automatically generates detailed statistics about the detection process, including:

- Detection confidence scores
- Facial expression distribution

These statistics are saved as charts and data files in the specified output directory.

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Matplotlib (for statistics visualization)
- YOLOv8

## Installation

```shellscript
# Clone the repository
git clone https://github.com/yourusername/video-object-detection.git

# Install dependencies
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
