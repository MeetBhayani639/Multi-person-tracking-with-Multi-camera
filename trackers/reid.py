"""
trackers/reid.py
Provides a `FeatureExtractor` class with two modes:
- 'torch': uses a supplied PyTorch model (expects model(input_tensor) -> embedding)
- 'hist': fallback RGB histogram-based descriptor (fast, no dependencies)

Set mode='torch' if you have a pretrained ReID model (OSNet/resnet) and provide the model instance.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(self, mode='hist', device='cpu', model=None, input_size=(128, 256)):
        assert mode in ('hist', 'torch')
        self.mode = mode
        self.device = device
        self.model = model
        self.input_size = input_size
        if self.model is not None:
            self.model.to(device)
            self.model.eval()

    def preprocess_torch(self, img):
        # img is a HxW x BGR np.array; convert to RGB, resize, to tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size[0], self.input_size[1]))
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    def extract(self, img) -> np.ndarray:
        """
        Input: BGR crop (numpy)
        Output: 1D L2-normalized vector (numpy float32)
        """
        if self.mode == 'torch' and self.model is not None:
            with torch.no_grad():
                t = self.preprocess_torch(img)
                emb = self.model(t)  # assume the model returns features of shape (1, D)
                if isinstance(emb, (tuple, list)):
                    emb = emb[0]
                emb = emb.squeeze(0)
                emb = F.normalize(emb, p=2, dim=0).cpu().numpy()
                return emb.astype(np.float32)
        # histogram fallback
        # convert to HSV, compute concatenated hist on H and S channels
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist = np.concatenate([h_hist.flatten(), s_hist.flatten()])
        hist = hist / (np.linalg.norm(hist) + 1e-6)
        return hist.astype(np.float32)
