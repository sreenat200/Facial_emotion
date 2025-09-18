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

# Streamlit app
st.title("📷 Live Facial Emotion Detection")

st.write("Use the camera below to capture your face and detect emotions:")

# Camera input
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Read image from buffer
    image = Image.open(img_file_buffer).convert("L")  # convert to grayscale
    image_resized = image.resize((48, 48))
    
    # Preprocess
    face_tensor = transform_live(image_resized).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(face_tensor)
        _, pred = torch.max(output, 1)
        emotion = emotions[pred.item()]

    # Show result
    st.image(image, caption=f"Detected Emotion: **{emotion}**", use_column_width=True)
    st.success(f"😊 Emotion detected: **{emotion}**")
