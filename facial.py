import cv2
import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import numpy as np
from huggingface_hub import PyTorchModelHubMixin

# Define SimpleCNN architecture with PyTorchModelHubMixin
class SimpleCNN(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
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

# Load trained model from Hugging Face
model = SimpleCNN.from_pretrained("sreenathsree1578/facial_emotion")
model.eval()

# Transform for live feed
transform_live = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Global variables
camera_on = False
cap = None
frame_placeholder = st.empty()

# Function to start/stop camera
def toggle_camera():
    global camera_on, cap
    if not camera_on:
        cap = cv2.VideoCapture(0)
        camera_on = True
    else:
        if cap is not None:
            cap.release()
        camera_on = False

# Streamlit app
st.title("Live Facial Emotion Detection")

# Buttons
if st.button("Start Emotion Detection"):
    toggle_camera()

if st.button("Stop Emotion Detection"):
    toggle_camera()

# Display video feed
while camera_on and cap is not None:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = transform_live(Image.fromarray(face)).unsqueeze(0)
            with torch.no_grad():
                output = model(face)
                _, pred = torch.max(output, 1)
                emotion = emotions[pred.item()]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
    else:
        break

# Cleanup on app close
if cap is not None and not cap.isOpened():
    cap.release()
