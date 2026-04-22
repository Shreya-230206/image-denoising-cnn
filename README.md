# Image Restoration under Network Distortions

## Overview
This project focuses on restoring images degraded by network-related distortions such as compression, blur, and packet loss, along with adversarial noise. A Convolutional Neural Network (CNN) is trained to reconstruct high-quality images from degraded inputs.

---

## Objective
To develop a deep learning model that enhances and restores images affected by:
- Network compression artifacts
- Blur and noise distortion
- Adversarial perturbations

---

## Methodology
1. Load dataset (CIFAR images)
2. Simulate degradation:
   - Noise injection
   - Blur/compression effects
3. Train CNN model to map:
   **Degraded Image → Clean Image**
4. Evaluate performance using loss reduction

---

## Tech Stack
- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Shreya-230206/image-denoising-cnn.git
cd image-denoising-cnn
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the model
```bash
python train.py
```
Expected Outcome:
```bash
Files already downloaded and verified
Epoch 1, Loss: 0.0038
Epoch 2, Loss: 0.0029
...
```
### 4. Test the model
```bash
python test.py
```
Expected Output:
Displays restored images and Shows comparison between:
- Original image
- Noisy/degraded image
- Restored output
