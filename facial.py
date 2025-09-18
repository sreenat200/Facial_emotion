import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Define SimpleCNN architecture for facial_emotion (grayscale, in_channels=1)
class SimpleCNN(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=7, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128 * 6 * 6, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Define EmotionDetectionCNN for emotion_detection (RGB, in_channels=3)
class EmotionDetectionCNN(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=7, in_channels=3):
        super(EmotionDetectionCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128 * 6 * 6, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Transform for image processing
def get_transform(in_channels):
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels)
    ])

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Streamlit app
st.markdown("<h3>Live Facial Emotion Detection</h3>", unsafe_allow_html=True)

# Sidebar for model, quality, and FPS selection with highest values as default
with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Model",
        ["sreenathsree1578/facial_emotion", "sreenathsree1578/emotion_detection"],
        index=1  # Default to emotion_detection
    )
    quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"], index=2)  # Default to High
    fps = st.selectbox("Select FPS", [15, 30, 60], index=2)  # Default to 60

# Map quality to resolution
quality_map = {
    "High (1080p)": {"width": 1920, "height": 1080},
    "Low (480p)": {"width": 854, "height": 480},
    "Medium (720p)": {"width": 1280, "height": 720}
}
resolution = quality_map[quality]

@st.cache_resource
def load_facial_emotion_model():
    try:
        config_path = hf_hub_download(repo_id="sreenathsree1578/facial_emotion", filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        num_classes = config.get("num_classes", 7)
        model = SimpleCNN(num_classes=num_classes, in_channels=1)
        model = model.from_pretrained("sreenathsree1578/facial_emotion")
        model.eval()
        return model, 1
    except Exception as e:
        st.error(f"Error loading facial_emotion: {str(e)}. Using default.")
        return SimpleCNN(num_classes=7, in_channels=1), 1

@st.cache_resource
def load_emotion_detection_model():
    try:
        config_path = hf_hub_download(repo_id="sreenathsree1578/emotion_detection", filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        num_classes = config.get("num_classes", 7)
        model = EmotionDetectionCNN(num_classes=num_classes, in_channels=3)
        model = model.from_pretrained("sreenathsree1578/emotion_detection")
        model.eval()
        return model, 3
    except Exception as e:
        st.error(f"Error loading emotion_detection: {str(e)}. Using default.")
        return SimpleCNN(num_classes=7, in_channels=1).from_pretrained("sreenathsree1578/facial_emotion"), 1

# Load selected model
if model_option == "sreenathsree1578/facial_emotion":
    model, in_channels = load_facial_emotion_model()
else:
    model, in_channels = load_emotion_detection_model()
transform_live = get_transform(in_channels)

# Emotion color mapping
emotion_colors = {
    'angry': (0, 0, 255),    # Red
    'disgust': (0, 255, 0),  # Green
    'fear': (255, 0, 0),     # Blue (default)
    'happy': (255, 0, 0),    # Blue (default)
    'sad': (0, 165, 255),    # Orange
    'surprise': (255, 0, 0), # Blue (default)
    'neutral': (255, 0, 0)   # Blue (default)
}

# Custom VideoProcessor for streamlit-webrtc
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w] if in_channels == 3 else gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = transform_live(Image.fromarray(face if in_channels == 3 else face, mode='RGB' if in_channels == 3 else 'L')).unsqueeze(0)
            with torch.no_grad():
                output = model(face)
                _, pred = torch.max(output, 1)
                emotion = emotions[pred.item()] if pred.item() < len(emotions) else "unknown"
            # Draw rectangle and text with emotion-specific color
            color = emotion_colors.get(emotion, (255, 0, 0))  # Default to blue if unknown
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            # Add white background for text
            text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(img, (x, y-35), (x+text_size[0], y-5), (255, 255, 255), -1)  # White background
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame.from_ndarray(img, format="bgr24")

# STUN configuration for WebRTC
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Auto-select the first available camera
def get_available_cameras():
    index = 0
    available_cameras = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)  # Use CAP_ANY for broader compatibility
        if not cap.isOpened():
            break
        available_cameras.append(index)
        cap.release()
        index += 1
    return available_cameras

# Get and use the first available camera
available_cameras = get_available_cameras()
if not available_cameras:
    st.error("No cameras found.")
else:
    selected_camera = available_cameras[0]  # Select the first available camera
    st.write(f"Using camera {selected_camera}")
    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": resolution["width"]},
                "height": {"ideal": resolution["height"]},
                "frameRate": {"ideal": fps},
                "deviceId": {"exact": selected_camera}  # Use the first available camera
            },
            "audio": False
        },
        async_processing=True,
        rtc_configuration=rtc_config
    )
