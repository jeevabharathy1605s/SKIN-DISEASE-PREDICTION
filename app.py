# ==========================================================
# 🩺 Skin Disease Classification with Grad-CAM
# Author: Jeeva Bharathy S
# ==========================================================

# ---------- Imports ----------
import os
import gdown
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------- Model Download ----------
MODEL_PATH = "efficientnet_skin_classifier.pth"

# 🧠 Replace below with your actual Google Drive FILE ID
GOOGLE_DRIVE_FILE_ID = "1xp9pGSoDGPJOTUHEccG__Hec2EEkRDiC"  # <--- put your real ID here

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive... please wait ⏳")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ---------- Device ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Model Definition ----------
class SkinClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("efficientnet_b4", pretrained=False)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ---------- Load Model ----------
num_classes = 6
model = SkinClassifier(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- Label Mappings ----------
label_to_idx = {
    "Actinic_Keratoses": 0,
    "Basal_Cell_Carcinoma": 1,
    "Benign_Keratosis": 2,
    "Dermatofibroma": 3,
    "Melanoma": 4,
    "Vascular_Lesions": 5
}
idx_to_label = {v: k for k, v in label_to_idx.items()}

# ---------- Image Preprocessing ----------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Skin Disease Classifier", layout="wide")
st.title("🩺 Skin Disease Classification with Grad-CAM")
st.write(
    "Upload a dermoscopic image to classify the disease and visualize model attention."
)

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

# ---------- Prediction Section ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    # Model Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
        pred_label = idx_to_label[pred_idx]
        confidence = torch.softmax(outputs, dim=1)[0][pred_idx].item()

    # Show result
    st.markdown(f"### 🧾 Prediction: **{pred_label}**")
    st.markdown(f"### 🔍 Confidence: **{confidence * 100:.2f}%**")

    # ---------- Grad-CAM ----------
    target_layers = [model.model.blocks[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    grayscale_cam = cam(
        input_tensor=input_tensor, targets=None, eigen_smooth=True, aug_smooth=True
    )
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # ---------- Display ----------
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)

    st.success("✅ Prediction complete!")

else:
    st.info("Please upload a skin image to begin prediction.")
