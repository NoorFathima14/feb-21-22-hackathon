# Team Members 
Noor Fathima (22PD26)

Sarnika Sanjeev Kumar (22PD31)

Sujan S (22PD35)


# Problem statement 3

# Build → Break → Improve: Navigating Synthetic Reality

> A hackathon project that builds a synthetic image detector, attacks it with adversarial perturbations, and hardens it — mirroring real-world AI security workflows.

---

## Problem Statement

> A photo of a crime scene. A viral image of a public figure. A piece of evidence in court. What if none of them were real?

The rapid advancement of generative AI has produced highly realistic synthetic images, creating serious challenges in cybersecurity, misinformation detection, and digital forensics. While detection models can perform well in controlled settings, many are surprisingly vulnerable to small, targeted modifications — an active arms race between synthesis and detection.

---

## Project Overview

This project follows a three-phase research-inspired cycle:

| Phase | Name | Goal |
|-------|------|------|
| 1 | **Build** | Train a CNN classifier to detect AI-generated images |
| 2 | **Break** | Use FGSM adversarial attacks to fool the detector |
| 3 | **Improve** | Retrain with adversarial examples to harden the model |

A **Streamlit web app** ties everything together — letting users upload any image and see live predictions, Grad-CAM visualizations, and side-by-side model comparisons.

---

## Project Structure

```
feb-21-22-hackathon/
│
├── data/
│   └── sample_images/           # Sample images for the Streamlit demo
│       ├── fake/
│       │   ├── fake_999(7).jpg
│       │   ├── fake_999(8).jpg
│       │   ├── fake_999(9).jpg
│       │   └── fake_999.jpg
│       └── real/
│           ├── real_0999(7).jpg
│           ├── real_0999(8).jpg
│           ├── real_0999(9).jpg
│           └── real_0999.jpg
│
├── outputs/                     # Pre-generated visualization outputs
│   ├── confusion_adv.png        # Adversarial model confusion matrix
│   ├── confusion_baseline.png   # Baseline model confusion matrix
│   └── robustness_curve.png     # Epsilon vs Accuracy curve
│
├── src/
│   ├── attacks/
│   │   ├── evaluate_fgsm.py     # Full adversarial evaluation across epsilons
│   │   ├── fgsm.py              # Core FGSM attack implementation
│   │   └── run_fgsm_experiment.py  # End-to-end attack experiment runner
│   │
│   ├── data/
│   │   ├── datasets.py          # DataLoader construction
│   │   ├── explore.py           # Dataset stats & visualization
│   │   └── transforms.py        # Preprocessing transforms
│   │
│   ├── evaluation/
│   │   ├── compare_models.py    # Baseline vs robust model comparison
│   │   ├── evaluate_adversarial.py  # Adversarial accuracy evaluation
│   │   └── evaluate_baseline.py    # Clean accuracy & metrics
│   │
│   ├── explainability/
│   │   ├── gradcam.py           # Grad-CAM implementation
│   │   └── visualize_gradcam.py # Heatmap overlay visualization
│   │
│   ├── models/
│   │   └── spatial_baseline.py  # CNN model architecture
│   │
│   └── utils/
│       ├── data_utils.py        # Shared data helpers
│       ├── model_utils.py       # Model save/load helpers
│       └── adversarial_train.py # Adversarial training loop
│
├── train.py                     # Main training entry point
├── app.py                       # Streamlit interactive demo
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore
```

---

## Dataset

**CIFAKE – Real and AI-Generated Synthetic Images**  
Source: [Kaggle — birdy654/cifake](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

| Split | FAKE | REAL | Total |
|-------|------|------|-------|
| Train | 50,000 | 50,000 | 100,000 |
| Test  | 10,000 | 10,000 | 20,000 |

- **Image size:** 32×32 RGB pixels
- **Real images:** From CIFAR-10 (photographs)
- **Fake images:** AI-generated using Stable Diffusion to match CIFAR-10 categories
- **Why CIFAKE?** Fast training (minutes per epoch), low memory footprint, perfect for from-scratch experiments and evasion research

### Preprocessing

All images are normalized to `[-1, 1]` range:
```python
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```
This stabilizes CNN training and is standard for image classification tasks.

---

## Phase 1: Build — Synthetic Image Detector

### Model Architecture: `SpatialBaselineCNN`

A custom 3-block convolutional neural network built from scratch:

```
Input: 3 × 32 × 32
    ↓
Block 1: Conv2d(3→32, 3×3) + BatchNorm + ReLU + MaxPool2d(2×2)   → 32×16×16
    ↓
Block 2: Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool2d(2×2)  → 64×8×8
    ↓
Block 3: Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool2d(2×2) → 128×4×4
    ↓
Flatten → FC(2048→256) + ReLU + Dropout(0.5)
    ↓
FC(256→2) → Logits [FAKE / REAL]
```

**Design Choices:**
- **BatchNorm** after each conv layer for training stability
- **MaxPooling** halves spatial dimensions each block: 32→16→8→4
- **Dropout (0.5)** before final layer to prevent overfitting
- **CrossEntropyLoss** + **Adam (lr=0.001)**

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Hardware | CPU (Kaggle) |

### Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.2500 | 89.85% | 0.1862 | 92.73% |
| 2 | 0.1795 | 93.14% | 0.1569 | 93.90% |
| 3 | 0.1593 | 94.00% | 0.1711 | 93.44% |
| 4 | 0.1437 | 94.60% | 0.1381 | 94.87% |
| 5 | 0.1287 | 95.22% | 0.1502 | 94.28% |

### Evaluation on Test Set

**Classification Report (Baseline Model):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| FAKE  | 0.92      | 0.97   | 0.94     | 10,000  |
| REAL  | 0.96      | 0.92   | 0.94     | 10,000  |
| **Overall** | **0.94** | **0.94** | **0.94** | **20,000** |

**Confusion Matrix (Baseline):**

<img width="518" height="437" alt="image" src="https://github.com/user-attachments/assets/f8292f70-4340-47fb-a80f-4b454d32422c" />


**Clean Accuracy: 94.99%**

### Visual Explainability: Grad-CAM


**What is Grad-CAM?**

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions of an image drive the model's decision.


**How it works:**
1. Compute gradient of the predicted class score with respect to the final conv layer's feature maps
2. Average gradients across spatial dimensions to get importance weights
3. Take a weighted sum of feature maps → apply ReLU
4. Overlay the resulting heatmap on the original image

In our baseline model, Grad-CAM reveals that the model heavily focuses on **texture patterns and global color statistics** — particularly the **upper half (sky regions)** of images. This concentration on specific spatial zones rather than structured semantic features is what makes it exploitable.

---

## Phase 2: Break — Adversarial Attacks (FGSM)

### Attack Method: Fast Gradient Sign Method (FGSM)

FGSM generates adversarial perturbations by computing the gradient of the loss with respect to the input image and then taking a small step in the direction that maximizes the loss.

**Mathematical formulation:**
```
x_adv = x + ε · sign(∇ₓ J(θ, x, y))
```

Where:
- `x` = original image
- `ε` = perturbation strength (epsilon)
- `∇ₓ J` = gradient of the loss w.r.t. the input image
- `sign()` = sign function returning +1 or -1 per pixel

**Key property:** The perturbation is nearly invisible to humans but specifically crafted to mislead the model.

### Finding a Target Image

We selected a FAKE image that the baseline model classified with **99.9999% confidence** as FAKE (index 0 in the test set — an airplane-like synthetic image).

### Iterative Evasion Results

| Epsilon (ε) | Prediction | Confidence |
|-------------|------------|------------|
| 0.00 (original) | FAKE | 99.9999% |
| 0.01 | FAKE | ~high |
| 0.03 | FAKE → borderline | dropping |
| 0.05 | REAL | ~low |
| 0.07 | REAL | moderate |
| 0.10 | REAL | high |

At **ε = 0.04**, the baseline model's prediction flips from **FAKE → REAL** — meaning the attack succeeded with a very small perturbation that remains visually imperceptible to a human observer.

### Full Adversarial Robustness Evaluation

Evaluated across the entire 20,000-image test set:

| Epsilon (ε) | Baseline Accuracy | Notes |
|-------------|------------------|-------|
| 0.01 | 80.12% | Slight degradation |
| 0.03 | 42.41% | Model nearly random |
| 0.05 | 19.80% | Worse than random |
| 0.07 | 11.89% | Almost completely fooled |
| 0.10 | 8.93%  | Model essentially fails |

### Why the Evasion Worked

**Grad-CAM comparison (original vs adversarial):**

- **Original Grad-CAM:** The model focused on the upper regions (sky/background texture) with a diffuse heatmap spread across the image
- **Adversarial Grad-CAM:** After FGSM, the attention map becomes nearly uniform/blank — the perturbation systematically suppressed the spatial attention signals the model relied on

**Root cause analysis:**

The baseline model learned to detect AI-generated images primarily by recognizing **high-frequency texture artifacts** — the subtle patterns that diffusion models leave in pixel space. FGSM directly attacked these gradients, injecting noise that disguised those texture patterns. Since the model lacked any multi-scale or frequency-aware representations, a tiny ε was sufficient to destroy its confidence.

---

## Phase 3: Improve — Adversarial Training

### Strategy: FGSM Adversarial Training

The hardened model is trained on a mix of clean and adversarially perturbed images simultaneously.

**Training loop per batch:**
1. Compute gradients from clean images
2. Generate FGSM adversarial examples at **ε = 0.03**
3. Compute loss on both clean and adversarial images
4. Backpropagate the **average combined loss**

```python
loss_total = (loss_clean + loss_adv) / 2
```

### Robust Model Architecture: `RobustCNN`

Structurally similar to the baseline but with:
- **No BatchNorm** in conv layers (reduces gradient signal leakage)
- **Dropout (0.5)** moved to classifier head for better regularization
- Separated `features` and `classifier` blocks for cleaner design

### Adversarial Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.4332 | 71.13% | 0.2248 | 91.49% |
| 2 | 0.3363 | 78.04% | 0.1953 | 92.26% |
| 3 | 0.3034 | 80.17% | 0.1883 | 93.42% |
| 4 | 0.2780 | 81.80% | 0.1704 | 93.28% |
| 5 | 0.2568 | 83.00% | 0.1647 | 93.81% |

> Note: Higher train loss is expected — the model is simultaneously learning to handle both clean and perturbed inputs.

---

## Results & Metrics

### Clean Accuracy Comparison

| Model | Clean Accuracy |
|-------|---------------|
| Baseline | **94.99%** |
| Robust (Adversarially Trained) | 93.66% |

> The robust model trades ~1.3% clean accuracy for dramatically improved adversarial resilience.

### Adversarial Accuracy Comparison

| Epsilon (ε) | Baseline | Robust Model | Improvement |
|-------------|----------|--------------|-------------|
| 0.01 | 82.08% | **90.21%** | +8.13% |
| 0.03 | 43.92% | **80.55%** | +36.63% |
| 0.05 | 21.27% | **68.54%** | +47.27% |
| 0.07 | 13.34% | **55.77%** | +42.43% |
| 0.10 | 9.58%  | **37.84%** | +28.26% |

### Confusion Matrices

**Baseline Model (Clean Test Set):**

<img width="518" height="437" alt="image" src="https://github.com/user-attachments/assets/8bca33f7-29d4-4c1f-8472-378afcafae5a" />


**Robust Model (Clean Test Set):**

<img width="524" height="433" alt="image" src="https://github.com/user-attachments/assets/871b0f0e-6b4c-4184-a74a-50380f3b6d68" />

### Adversarial Robustness Curve

The robustness curve (Epsilon vs Accuracy) clearly shows:
- **Baseline:** Plummets from 82% → 9% as ε increases from 0.01 → 0.10
- **Robust model:** Degrades more gracefully, maintaining 90% at ε=0.01 and 38% at ε=0.10

---

## Interactive Demo (Streamlit App)

The `app.py` Streamlit application provides a full interactive interface with four distinct sections.

### Running the App

```bash
streamlit run app.py
```

Both `baseline_model.pth` and `adversarial_model.pth` must be present in the `models/` directory. Models are cached with `@st.cache_resource` for fast reloads.

### App Sections

**1. Dataset Samples**

Loads images from `data/sample_images/fake/` and `data/sample_images/real/` and displays them in a 4-column grid. Shows FAKE and REAL samples side-by-side so users can visually compare AI-generated vs real images.

**2. Upload Image for Testing**

- Upload any PNG/JPG/JPEG image
- Image is resized to **32×32**, normalized to `[-1, 1]`, and passed through the **Baseline Model**
- Shows: predicted class (FAKE/REAL) + raw confidence score
- Generates a **Grad-CAM heatmap** using `model.conv3` as the target layer (registered via forward + full backward hooks)
- Heatmap is overlaid on the original image using `cv2.COLORMAP_JET` with 60/40 blending

**3. FGSM Attack**

- An **epsilon slider** (0.00 → 0.10, default 0.03) lets users control attack strength
- On slider change, `fgsm_attack()` runs against the **Baseline Model** using the uploaded image
- Shows the perturbed image and the **Baseline Model's new prediction** after attack
- Also runs the same attacked image through the **Robust Model** to show the comparison
- Side-by-side **Grad-CAM comparison**: Baseline Grad-CAM vs Robust Grad-CAM on the attacked image — visually shows how attention maps shift under perturbation

**4. Model Metrics & Comparison**

Displays precomputed evaluation results (hardcoded from full test-set evaluation):

```python
clean_baseline = 0.9499
clean_robust   = 0.9366

epsilons           = [0.01, 0.03, 0.05, 0.07, 0.1]
baseline_adv_acc   = [0.8208, 0.4392, 0.2127, 0.1334, 0.0958]
robust_adv_acc     = [0.9021, 0.8055, 0.6854, 0.5577, 0.3784]
```

- **`st.metric`** widgets for clean accuracy comparison
- **`st.dataframe`** for the full adversarial accuracy table
- Loads pre-saved PNG files from `outputs/`:
  - `confusion_baseline.png`
  - `confusion_adv.png`
  - `robustness_curve.png`
- Matplotlib chart of the Adversarial Robustness Comparison (Baseline vs Robust curves)

### Grad-CAM Implementation (in app.py)

The app uses hook-based Grad-CAM rather than the standalone `src/explainability/gradcam.py`, allowing it to work inline with model inference:

```python
# Forward hook captures feature map activations
handle_f = model.conv3.register_forward_hook(forward_hook)

# Backward hook captures gradients
handle_b = model.conv3.register_full_backward_hook(backward_hook)

# CAM = ReLU(weighted sum of activation maps)
weights = torch.mean(grads, dim=[2, 3], keepdim=True)
cam = F.relu(torch.sum(weights * acts, dim=1))
```

Hooks are properly removed after each call to avoid accumulation across slider interactions.

---

## Output

![WhatsApp Image 2026-02-21 at 10 10 15 PM (1)](https://github.com/user-attachments/assets/4722ded0-1f78-4deb-9dd0-9a1d1b0ebeb7)
![WhatsApp Image 2026-02-21 at 10 10 15 PM (2)](https://github.com/user-attachments/assets/09319895-7c85-48e1-91b4-3a3970eefef3)
![WhatsApp Image 2026-02-21 at 10 10 15 PM](https://github.com/user-attachments/assets/9b53198d-3720-42e5-ba77-32454c4400ed)
![WhatsApp Image 2026-02-21 at 10 10 16 PM (1)](https://github.com/user-attachments/assets/e1d90650-90ed-46d1-9cf5-449611333cf7)
![WhatsApp Image 2026-02-21 at 10 10 16 PM](https://github.com/user-attachments/assets/7185fd05-c663-4619-9781-18e061dacd40)


## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

**Key dependencies:**
```
torch
torchvision
streamlit
matplotlib
seaborn
scikit-learn
numpy
opencv-python
tqdm
pandas
pillow
```

### Option 1: Run on Kaggle (Recommended for Training)

1. Upload the notebook to Kaggle
2. Add the CIFAKE dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
3. Enable GPU accelerator
4. Run training cells → models save to `models/baseline_model.pth` and `models/adversarial_model.pth`

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/NoorFathima14/feb-21-22-hackathon.git
cd feb-21-22-hackathon

# Install dependencies
pip install -r requirements.txt

# Train baseline model (requires CIFAKE dataset)
python train.py

# Launch Streamlit app (models must exist in models/)
streamlit run app.py
```

### Option 3: Skip Training — Just Run the App

If you already have the `.pth` model files:

```bash
# Place models in the models/ directory
# Place sample images in data/sample_images/fake/ and data/sample_images/real/
# Place output PNGs in outputs/

streamlit run app.py
```

### Expected Compute

| Task | Time (CPU) | Time (GPU) |
|------|-----------|-----------|
| 1 epoch baseline training | ~6 min | ~45 sec |
| 5 epoch baseline training | ~25 min | ~4 min |
| Full adversarial evaluation (per ε) | ~6 min | ~1 min |
| Adversarial training (5 epochs) | ~30 min | ~5 min |

---

## Key Findings & Analysis

### What the Baseline Model Actually Learned

From Grad-CAM analysis, the baseline CNN focused on:
- **High-frequency texture artifacts** specific to diffusion model outputs
- **Background/sky regions** rather than semantic object features
- **Global color distribution patterns**

This is a classic sign of a model that learned *statistical shortcuts* rather than *semantic understanding*.

### Why FGSM Was So Effective

The FGSM attack works because:
1. **Gradient-based:** It directly uses the model's own decision boundary gradients
2. **High-frequency targeting:** It injects noise that overlaps with the exact frequency bands the model relied on
3. **Small ε sufficiency:** Since the model's features were brittle texture cues, tiny ε values (~0.03–0.05) were enough to destroy them

At ε=0.04, a pixel shift of roughly 1/255 in normalized space was sufficient to flip predictions — demonstrating just how fragile texture-based detection is.

### Why Adversarial Training Helped

By exposing the model to FGSM examples during training:
- The model learned to classify correctly **even when texture artifacts are perturbed**
- It was forced to find more **robust, distributed features**
- Grad-CAM of the robust model shows **broader, more semantically meaningful attention** — focusing on the actual object structure rather than background noise

### Limitations & Future Work

| Limitation | Proposed Solution |
|------------|------------------|
| Only FGSM attacks tested | Add PGD, C&W, AutoAttack |
| Single epsilon for adversarial training | Multi-epsilon curriculum training |
| Spatial-only model | Add frequency-domain branch (DCT features) |
| 32×32 images only | Test on higher resolution synthetic images |
| No data augmentation during training | Add random crops, flips, color jitter |
| FGSM is white-box (requires gradients) | Evaluate on black-box transfer attacks |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Computer Vision | torchvision, OpenCV (`cv2`) |
| Data Processing | NumPy, Pandas, PIL |
| Visualization | Matplotlib, Seaborn |
| Explainability | Grad-CAM (hook-based, custom implementation) |
| Metrics | scikit-learn |
| Web App | Streamlit |
| Training Platform | Kaggle (CPU) |
| Dataset | CIFAKE (Kaggle) |

