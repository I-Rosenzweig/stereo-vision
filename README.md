# stereo-vision


# Stereo Vision Object Detection System

A real-time stereo vision system that combines depth perception with YOLOv8 object detection for accurate distance estimation and object recognition.

## Features

- Real-time stereo vision processing using two synchronized camera feeds
- Object detection using YOLOv8 neural network
- Distance estimation for detected objects
- Depth map visualization using WLS filtering
- Radar integration for additional spatial awareness
- Video recording with timestamp-based file naming
- Multi-threaded design for efficient stream handling

![image](https://github.com/user-attachments/assets/e9e53d53-4d5a-4639-a22d-4cede142c231)


## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Ultralytics YOLOv8 (`pip install ultralytics`)
- NumPy
- Socket library (built-in)
- Threading library (built-in)

## Hardware Requirements

- Two synchronized cameras (stereo setup)
- Optional: Radar sensor
- Computing device capable of running real-time ML inference

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install ultralytics opencv-python numpy
```

3. Download the YOLOv8 model:
```bash
# The script will automatically download yolov8n.pt on first run
# Or manually place your custom YOLOv8 model in the project directory
```

## Configuration

1. Update the host IP address in `stereo_vision_final.py`:
```python
host = "192.168.137.94"  # Replace with your server IP
```

2. Adjust the camera parameters if needed:
```python
baseline = 0.45  # Distance between cameras in meters
focal_length = 650  # Camera focal length in mm
fx = 600  # Focal length in pixels
```

## Usage

1. Start the server script on your camera system (not included in this repository).

2. Run the main script:
```bash
python stereo_vision_final.py
```

3. The system will display:
- Left camera feed with object detection boxes and distance estimates
- Depth map visualization
- Radar data in the console (if connected)

4. Press 'q' to quit the application.

## Output

- Real-time visualization of:
  - Object detection with bounding boxes
  - Distance measurements
  - Confidence scores
  - Color-coded depth map
- Video recording saved with timestamp (format: `recorded_stream_DD-MM-YYYY_HH-MM-SS.mp4`)

## System Architecture

- Multi-threaded design with separate threads for:
  - Left camera stream processing
  - Right camera stream processing
  - Radar data processing
  - Main processing loop (depth mapping and object detection)

## Limitations

- Requires proper camera calibration for accurate distance estimation
- Performance depends on hardware capabilities
- YOLOv8 detection accuracy may vary based on lighting conditions and object size

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Your chosen license]

## Contact

[Your contact information]
