# Real-Time Facial Expression Recognition with YOLOv8

A YOLOv8-based system for detecting facial expressions in video content. Provides real-time bounding box annotations and generates temporal statistics about emotional patterns.

## Key Features
- **6 Core Expression Classes**: Angry 😠, Happy 😄, Sad 😢, Surprise 😲, Fear 😨, Disgust 🤢
- **Visual Annotations**: Real-time bounding boxes with expression labels and confidence scores
- **Temporal Analysis**: Charts showing emotion distribution and confidence trends over time
- **Performance Tracking**: FPS metrics and detection consistency reports

## Example Results

### Case Study 1: Angry Expression Analysis
**Input Video**  
![Angry Detection Demo](demo/demo_angry/messi_angry_with_detections.gif)

**Statistical Output**  
![Angry Statistics](demo/demo_angry/demo_stats_angry.png)
- *Left*: Dominant expression timeline visualization
- *Middle*: Confidence score distribution across detections
- *Right*: Total Facial Expression counts

### Case Study 2: Happy Expression Analysis
**Input Video**  
![Happy Detection Demo](demo/demo_happy/messi_happy_with_detections.gif)

**Statistical Output**  
![Happy Statistics](demo/demo_happy/demo_stats_happy.png)
- *Left*: Dominant expression timeline visualization
- *Middle*: Confidence score distribution across detections
- *Right*: Total Facial Expression counts

## Installation

```bash
git clone https://github.com/yourusername/facial-expression-recognition.git
cd facial-expression-recognition
pip install -r requirements.txt
```

## Usage

```bash
# Basic video analysis with webcam feed
python main.py --camera 0

# Process video file with statistics generation
python main.py --video input.mp4 --stats

# Save annotated output to specific path
python main.py --video input.mp4 --output annotated_video.mp4

# Use specific detector backend with custom stats directory
python main.py --video input.mp4 --detector-backend yolov8 --stats --stats-dir ./analytics
```
