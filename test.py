import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Enhancer
from data_pipeline import build_dataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

os.makedirs("results", exist_ok=True)

model = Enhancer()
model.load_state_dict(torch.load("enhancer.pth", map_location="cpu"))
model.eval()

X, Y = build_dataset(10)

with torch.no_grad():
    for i in range(10):
        inp = X[i].unsqueeze(0)
        output = model(inp)[0]

        degraded = X[i].permute(1,2,0).cpu().numpy()
        original = Y[i].permute(1,2,0).cpu().numpy()
        restored = output.detach().permute(1,2,0).cpu().numpy()

        print("PSNR:", psnr(original, restored))
        print("SSIM:", ssim(original, restored, channel_axis=2, data_range=1.0))

        plt.figure(figsize=(10,3))

        plt.subplot(1,3,1)
        plt.title("Original")
        plt.imshow(original)

        plt.subplot(1,3,2)
        plt.title("Degraded")
        plt.imshow(degraded)

        plt.subplot(1,3,3)
        plt.title("Restored")
        plt.imshow(restored)

        plt.show()

        baseline = cv2.GaussianBlur(degraded, (5,5), 0)

        print("Baseline PSNR:", psnr(original, baseline))
        print("CNN PSNR:", psnr(original, restored))

        cv2.imwrite("results/original.png", (original*255).astype('uint8'))
        cv2.imwrite("results/degraded.png", (degraded*255).astype('uint8'))
        cv2.imwrite("results/restored.png", (restored*255).astype('uint8'))