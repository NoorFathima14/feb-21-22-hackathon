import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

from src.models.spatial_baseline import SpatialBaselineCNN

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Adversarial Robustness Demo",
    layout="centered"
)


st.title("Adversarial Robustness in Synthetic Image Detection")
st.markdown("Baseline CNN vs Adversarially Trained CNN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load Models
# -------------------------------------------------
@st.cache_resource
def load_models():

    baseline = SpatialBaselineCNN().to(device)
    baseline.load_state_dict(
        torch.load("models/baseline_model.pth", map_location=device)
    )
    baseline.eval()

    robust = SpatialBaselineCNN().to(device)
    robust.load_state_dict(
        torch.load("models/adversarial_model.pth", map_location=device)
    )
    robust.eval()

    return baseline, robust


baseline_model, robust_model = load_models()

# ==============================
# Precomputed Evaluation Results
# ==============================

clean_baseline = 0.9499
clean_robust = 0.9366

epsilons = [0.01, 0.03, 0.05, 0.07, 0.1]

baseline_adv_acc = [0.8208, 0.4392, 0.2127, 0.1334, 0.0958]
robust_adv_acc = [0.9021, 0.8055, 0.6854, 0.5577, 0.3784]

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess(image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = (image - 0.5) / 0.5
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    return image.to(device)


# -------------------------------------------------
# FGSM Attack
# -------------------------------------------------
def fgsm_attack(model, image, epsilon, label):

    image.requires_grad = True

    output = model(image)
    loss = F.cross_entropy(output, label)

    model.zero_grad()
    loss.backward()

    perturbed = image + epsilon * image.grad.sign()
    perturbed = torch.clamp(perturbed, -1, 1)

    return perturbed.detach()


# -------------------------------------------------
# Grad-CAM
# -------------------------------------------------
def generate_gradcam(model, image_tensor, target_class):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = model.conv3.register_forward_hook(forward_hook)
    handle_b = model.conv3.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    loss = output[:, target_class]

    model.zero_grad()
    loss.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * acts, dim=1)

    cam = F.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    cam = cv2.resize(cam, (32, 32))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_f.remove()
    handle_b.remove()

    return cam


# -------------------------------------------------
# Dataset Samples Section
# -------------------------------------------------
st.header("Dataset Samples")

sample_root = "data/sample_images"

if os.path.exists(sample_root):

    fake_dir = os.path.join(sample_root, "fake")
    real_dir = os.path.join(sample_root, "real")

    st.subheader("FAKE Samples")
    if os.path.exists(fake_dir):
        fake_images = os.listdir(fake_dir)[:4]
        cols = st.columns(len(fake_images))

        for col, img_name in zip(cols, fake_images):
            img = Image.open(os.path.join(fake_dir, img_name))
            col.image(img, caption="FAKE", use_column_width=True)

    st.subheader("REAL Samples")
    if os.path.exists(real_dir):
        real_images = os.listdir(real_dir)[:4]
        cols = st.columns(len(real_images))

        for col, img_name in zip(cols, real_images):
            img = Image.open(os.path.join(real_dir, img_name))
            col.image(img, caption="REAL", use_column_width=True)

else:
    st.info("Sample images folder not found.")
# -------------------------------------------------
# Upload Section
# -------------------------------------------------
st.header("Upload Image for Testing")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", width=250)

    input_tensor = preprocess(image)

    # ---------------------------------------------
    # Baseline Prediction
    # ---------------------------------------------
    baseline_output = baseline_model(input_tensor)
    baseline_prob = F.softmax(baseline_output, dim=1)
    baseline_pred = torch.argmax(baseline_prob, dim=1)

    st.subheader("Baseline Model Prediction")
    st.write("Prediction:", "REAL" if baseline_pred.item() == 1 else "FAKE")
    confidence = torch.max(baseline_prob).detach().cpu().item()
    st.write("Confidence:", confidence)
    # ---------------------------------------------
    # Baseline Grad-CAM
    # ---------------------------------------------
    st.subheader("Grad-CAM (Baseline)")

    cam = generate_gradcam(
        baseline_model,
        input_tensor.clone(),
        baseline_pred.item()
    )

    original_img = np.array(image.resize((32, 32))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0

    overlay = 0.6 * original_img + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)
    st.image(overlay, width=250)

    # ---------------------------------------------
    # FGSM Attack
    # ---------------------------------------------
    st.subheader("FGSM Attack")

    epsilon = st.slider("Epsilon", 0.0, 0.1, 0.03)

    attacked = fgsm_attack(
        baseline_model,
        input_tensor.clone(),
        epsilon,
        baseline_pred
    )

    attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().numpy()
    attacked_img = attacked_img * 0.5 + 0.5

    st.image(attacked_img, caption="Adversarial Image", width=250)

    # Baseline on attacked
    attacked_output = baseline_model(attacked)
    attacked_pred = torch.argmax(attacked_output, dim=1)

    st.write("Baseline Prediction After Attack:",
             "REAL" if attacked_pred.item() == 1 else "FAKE")

    # ---------------------------------------------
    # Robust Model Prediction
    # ---------------------------------------------
    st.subheader("Robust Model on Attacked Image")

    robust_output = robust_model(attacked)
    robust_pred = torch.argmax(robust_output, dim=1)

    st.write("Prediction:",
             "REAL" if robust_pred.item() == 1 else "FAKE")

    # ---------------------------------------------
    # Grad-CAM Comparison
    # ---------------------------------------------
    st.subheader("Grad-CAM Comparison")

    cam_baseline_attacked = generate_gradcam(
        baseline_model,
        attacked.clone(),
        attacked_pred.item()
    )

    cam_robust = generate_gradcam(
        robust_model,
        attacked.clone(),
        robust_pred.item()
    )

    heatmap_base = cv2.applyColorMap(
        np.uint8(255 * cam_baseline_attacked),
        cv2.COLORMAP_JET
    ) / 255.0

    heatmap_rob = cv2.applyColorMap(
        np.uint8(255 * cam_robust),
        cv2.COLORMAP_JET
    ) / 255.0

    overlay_base = 0.6 * attacked_img + 0.4 * heatmap_base
    overlay_base = np.clip(overlay_base, 0, 1)
    overlay_rob = 0.6 * attacked_img + 0.4 * heatmap_rob
    overlay_rob = np.clip(overlay_rob, 0, 1)

    col1, col2 = st.columns(2)
    col1.image(overlay_base, caption="Baseline Grad-CAM", use_column_width=True)
    col2.image(overlay_rob, caption="Robust Grad-CAM", use_column_width=True)


# -------------------------------------------------
# Metrics Section
# -------------------------------------------------
st.markdown("---")
st.header("Model Metrics & Comparison")

col1, col2 = st.columns(2)

if os.path.exists("outputs/confusion_baseline.png"):
    col1.image("outputs/confusion_baseline.png",
               caption="Baseline Confusion Matrix")

if os.path.exists("outputs/confusion_adv.png"):
    col2.image("outputs/confusion_adv.png",
               caption="Adversarial Confusion Matrix")

if os.path.exists("outputs/robustness_curve.png"):
    st.image("outputs/robustness_curve.png",
             caption="Robustness Curve (Epsilon vs Accuracy)")
    
st.header("Clean Accuracy Comparison")

col1, col2 = st.columns(2)

col1.metric("Baseline Model", f"{clean_baseline:.4f}")
col2.metric("Robust Model", f"{clean_robust:.4f}")

import pandas as pd

st.header("Adversarial Accuracy Comparison")

adv_df = pd.DataFrame({
    "Epsilon": epsilons,
    "Baseline Accuracy": baseline_adv_acc,
    "Robust Accuracy": robust_adv_acc
})

st.dataframe(adv_df)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))  # 👈 smaller figure

ax.plot(epsilons, baseline_adv_acc, marker='o', label="Baseline")
ax.plot(epsilons, robust_adv_acc, marker='o', label="Robust")

ax.set_xlabel("Epsilon (FGSM Strength)")
ax.set_ylabel("Accuracy")
ax.set_title("Adversarial Robustness Comparison")
ax.legend()
ax.grid(True)

st.pyplot(fig)
