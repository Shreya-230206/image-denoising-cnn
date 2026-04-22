import numpy as np
import cv2
import torch
from torchvision import datasets, transforms

# --------- Load dataset ---------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)

# --------- Distortions ---------
def compress_image(img_tensor, quality=20):
    img = (img_tensor.permute(1,2,0).numpy()*255).astype(np.uint8)
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(enc, 1)
    return dec / 255.0

def packet_loss(img, prob=0.3):
    mask = np.random.rand(*img.shape) > prob
    return img * mask

def blur(img):
    return cv2.GaussianBlur(img, (5,5), 0)

def adversarial_noise(img):
    noise = np.random.normal(0, 0.05, img.shape)
    return np.clip(img + noise, 0, 1)

def degrade(img_tensor):
    img = compress_image(img_tensor)
    img = packet_loss(img)
    img = blur(img)
    img = adversarial_noise(img)
    return img

# --------- Build dataset pairs ---------
def build_dataset(n_samples=2000):
    X, Y = [], []
    for i in range(n_samples):
        img, _ = dataset[i]
        degraded = degrade(img)

        X.append(torch.tensor(degraded).permute(2,0,1).float())
        Y.append(img)

    return torch.stack(X), torch.stack(Y)