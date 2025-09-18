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
@st.cache_resource
def load_model():
    return SimpleCNN.from_pretrained("sreenathsree1578/facial_emotion")

model = load_model()
model.eval()

# Transform for live feed
transform_live = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Streamlit app
st.title("Live Facial Emotion Detection")
st.write("Click 'Start' to begin emotion detection and 'Stop' to end the session.")

# Initialize session state
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Function to start/stop camera
def toggle_camera():
    if not st.session_state.camera_on:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.camera_on = True
    else:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.camera_on = False

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Emotion Detection"):
        toggle_camera()

with col2:
    if st.button("Stop Emotion Detection"):
        toggle_camera()

# Display video feed
frame_placeholder = st.empty()
stop_button = st.button("Stop")

if st.session_state.camera_on and st.session_state.cap is not None:
    cap = st.session_state.cap
    while st.session_state.camera_on and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face_pil = Image.fromarray(face)
            face_tensor = transform_live(face_pil).unsqueeze(0)
            
            with torch.no_grad():
                output = model(face_tensor)
                _, pred = torch.max(output, 1)
                emotion = emotions[pred.item()]
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
        
        # Check if stop button was pressed
        if stop_button:
            toggle_camera()
            break

# Cleanup when stopping
if not st.session_state.camera_on and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None
    frame_placeholder.empty()
    st.write("Camera stopped.")

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a convolutional neural network (CNN) to detect emotions from facial expressions in real-time. "
    "The model was trained on the FER-2013 dataset and can recognize 7 different emotions: "
    "angry, disgust, fear, happy, sad, surprise, and neutral."
)
