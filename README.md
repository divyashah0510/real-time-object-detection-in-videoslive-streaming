# Real Time Object Detection using YoloV10 and OpenCV
This repository contains the code for real time object detection using YoloV10 and OpenCV. The code is written in Python and uses OpenCV to capture video frames from the webcam. The YoloV10 model is used to detect objects in the video frames. The detected objects are then drawn on the video frames along with the confidence scores.

## Requirements
- Python 3.6 or higher
- OpenCV
- Numpy
- YoloV10 weights and configuration file

## Installation
1. Clone the repository
```bash
git clone https://github.com/divyashah0510/real-time-object-detection-in-videoslive-streaming.git
```
2. Install the required packages
```bash
pip install -r requirements.txt
```
3. Download the YoloV10 weights and configuration file from the official Yolo website and place them in the `yolov10` directory.

## Usage
Run the following command to start real time object detection using YoloV10 and OpenCV.
```bash
python detect_objects.py
```
The video feed from the webcam will be displayed along with the detected objects and confidence scores.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Live Demo
You can see a live demo of the project [here](https://real-time-object-detection-in-video-live-streaming.streamlit.app/).