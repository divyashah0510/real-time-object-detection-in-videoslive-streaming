import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO('models/yolov10s.pt')

# YOLO class names
yolo_classes = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'TV monitor', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

st.set_page_config(page_icon="🔍", layout="wide", initial_sidebar_state="expanded", page_title="Object Detection")

# Initialize session state variables
if 'is_detecting' not in st.session_state:
    st.session_state.is_detecting = False
if 'is_webcam_active' not in st.session_state:
    st.session_state.is_webcam_active = False


# Function for live object detection using webcam
def live_streaming(conf_threshold, selected_classes):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while st.session_state.is_detecting and st.session_state.is_webcam_active:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf_threshold)
        detections = results[0]

        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        class_ids = detections.boxes.cls.cpu().numpy().astype(int)

        if selected_classes:
            filtered = [(box, conf, class_id) for box, conf, class_id in zip(boxes, confs, class_ids)
                        if yolo_classes[class_id] in selected_classes]
            if filtered:
                boxes, confs, class_ids = zip(*filtered)
            else:
                boxes, confs, class_ids = [], [], []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()


# Function for object detection on uploaded video
def video_streaming(uploaded_file, conf_threshold, selected_classes):
    stframe = st.empty()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and st.session_state.is_detecting:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf_threshold)
        detections = results[0]

        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        class_ids = detections.boxes.cls.cpu().numpy().astype(int)

        if selected_classes:
            filtered = [(box, conf, class_id) for box, conf, class_id in zip(boxes, confs, class_ids)
                        if yolo_classes[class_id] in selected_classes]
            if filtered:
                boxes, confs, class_ids = zip(*filtered)
            else:
                boxes, confs, class_ids = [], [], []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()


# Function for object detection on uploaded image
def image_detection(uploaded_file, conf_threshold, selected_classes):
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model.predict(source=image_cv, conf=conf_threshold)
    detections = results[0]

    boxes = detections.boxes.xyxy.cpu().numpy()
    confs = detections.boxes.conf.cpu().numpy()
    class_ids = detections.boxes.cls.cpu().numpy().astype(int)

    if selected_classes:
        filtered = [(box, conf, class_id) for box, conf, class_id in zip(boxes, confs, class_ids)
                    if yolo_classes[class_id] in selected_classes]
        if filtered:
            boxes, confs, class_ids = zip(*filtered)
        else:
            boxes, confs, class_ids = [], [], []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    st.image(image_cv, channels="BGR")


# Sidebar controls for user input
with st.sidebar:
    st.title("Object Detection Settings " + "⚙️")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3)
    selected_classes = st.multiselect("Select classes for object detection", yolo_classes)

    # Unified file uploader for both images and videos
    uploaded_file = st.file_uploader("Upload an image or video " + "📤",
                                     type=["mp4", "mov", "avi", "m4v", "jpg", "png", "jpeg"])

    if st.button("Use Webcam 📷" if not st.session_state.is_webcam_active else "Stop Webcam 🛑"):
        st.session_state.is_webcam_active = not st.session_state.is_webcam_active
        if st.session_state.is_webcam_active:
            st.session_state.is_detecting = True
        else:
            st.session_state.is_detecting = False

    detect_button = st.button("Start Detection ▶️" if not st.session_state.is_detecting else "Stop Detection 🛑",
                              disabled=(not uploaded_file and not st.session_state.is_webcam_active))

    if detect_button:
        st.session_state.is_detecting = not st.session_state.is_detecting

# Handle object detection based on user input
if st.session_state.is_detecting:
    if st.session_state.is_webcam_active:
        st.info("Detecting objects using webcam...")
        live_streaming(confidence_threshold, selected_classes)
    elif uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['mp4', 'mov', 'avi', 'm4v']:
            st.info("Detecting objects in video...")
            video_streaming(uploaded_file, confidence_threshold, selected_classes)
        elif file_extension in ['jpg', 'jpeg', 'png']:
            st.info("Detecting objects in image...")
            image_detection(uploaded_file, confidence_threshold, selected_classes)
else:
    st.title("Object Detection")
    st.info("Upload an image or video, or start the webcam for object detection.")

    st.write("""
        ### What is YOLO?
        YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system that excels in speed and accuracy. It processes images in a single pass, making it highly efficient for applications requiring rapid object detection.

        ### How YOLO Works
        YOLO divides the input image into a grid and predicts bounding boxes and class probabilities for each grid cell. This allows it to identify multiple objects simultaneously, making it suitable for real-time scenarios.

        ### Training YOLO
        To train a YOLO model on your own dataset, follow these key steps:

        1. **Dataset Preparation**:
            - Collect and annotate your images with bounding box coordinates and class labels. You can use annotation tools like LabelImg or Roboflow.

        2. **Environment Setup**:
            - Install the necessary libraries and dependencies as specified in the Ultralytics repository. This typically involves using Python and libraries like PyTorch.

        3. **Model Configuration**:
            - Choose a model architecture (e.g., YOLOv5) and configure it based on your dataset’s requirements. This includes setting the number of classes and adjusting the input image size.

        4. **Training**:
            - Use the command line interface to start the training process. The typical command looks like this:
              ```bash
              python train.py --img 640 --batch 16 --epochs 50 --data your_dataset.yaml --weights yolov5s.pt
              ```
            - Here, you specify parameters like image size, batch size, number of epochs, dataset configuration, and pre-trained weights.

        5. **Evaluation**:
            - After training, evaluate the model's performance using validation data. This step helps in understanding the accuracy and making necessary adjustments.

        For detailed instructions, examples, and best practices, please refer to the [Ultralytics YOLO Training Documentation](https://docs.ultralytics.com/modes/train/).

        Now, go ahead and upload your image or video, or start the webcam to see YOLO in action!
    """)
