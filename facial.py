import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Define SimpleCNN architecture with PyTorchModelHubMixin
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
st.title("Live Facial Emotion Detection")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["sreenathsree1578/facial_emotion", "sreenathsree1578/emotion_detection"]
)

@st.cache_resource
def load_model(model_name):
    try:
        # Get model configuration
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        # Determine input channels based on model name
        if model_name == "sreenathsree1578/emotion_detection":
            in_channels = 3  # RGB model
        else:
            in_channels = 1  # Grayscale model
            
        num_classes = config.get("num_classes", 7)
        
        # Load model with correct input channels
        model = SimpleCNN(num_classes=num_classes, in_channels=in_channels)
        
        # Load state dict
        model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Load state dict with strict=False to handle potential mismatches
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model, in_channels
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}. Using default.")
        # Fallback to default model
        model = SimpleCNN(num_classes=7, in_channels=1)
        return model, 1

# Load selected model
model, in_channels = load_model(model_option)
transform_live = get_transform(in_channels)

# Custom VideoProcessor for streamlit-webrtc
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract face region
            if in_channels == 3:
                # For RGB models, use color image
                face_roi = img[y:y+h, x:x+w]
                face_pil = Image.fromarray(face_roi, mode='RGB')
            else:
                # For grayscale models, use grayscale
                face_roi = gray[y:y+h, x:x+w]
                face_pil = Image.fromarray(face_roi, mode='L')
            
            # Transform and predict
            face_tensor = transform_live(face_pil).unsqueeze(0)
            
            with torch.no_grad():
                output = model(face_tensor)
                _, pred = torch.max(output, 1)
                emotion = emotions[pred.item()] if pred.item() < len(emotions) else "unknown"
            
            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# STUN configuration for WebRTC
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Start webcam stream with first camera
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
            "deviceId": {"exact": 0}
        },
        "audio": False
    },
    async_processing=True,
    rtc_configuration=rtc_config
)
