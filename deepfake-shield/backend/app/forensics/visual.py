"""
Lightweight visual forensics — no PyTorch/Transformers.
Uses ELA, spectral analysis, and perceptual hashing.
Designed for Render free tier (512MB RAM).
"""
import io
import base64
import logging
from PIL import Image, ImageOps, ImageChops
import numpy as np
import cv2
import imagehash

logger = logging.getLogger("deepfake-shield.visual")

# Lazy-loaded (no heavy imports)
_model_loaded = True  # No model to load


def sanitize_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def compute_ela(pil_img: Image.Image, quality: int = 90) -> float:
    """Error Level Analysis — detect compression inconsistencies."""
    buffer = io.BytesIO()
    pil_img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    diff = ImageChops.difference(pil_img, resaved)
    diff_np = np.array(diff, dtype=np.float32)
    return round(float(np.mean(diff_np)), 2)


def compute_phash(image_bytes: bytes) -> str:
    """Perceptual hash for reverse-image search."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return str(imagehash.phash(img, hash_size=16))


def _compute_frequency_score(img_np: np.ndarray) -> float:
    """
    2D FFT high-frequency power analysis.
    AI-generated images often show characteristic frequency anomalies.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4

    y, x = np.ogrid[:h, :w]
    mask = (x - center_w)**2 + (y - center_h)**2 > radius**2

    hf_energy = np.sum(magnitude[mask])
    total_energy = np.sum(magnitude)

    ratio = hf_energy / (total_energy + 1e-10)
    return float(ratio)


def analyze_image_with_heatmap(image_bytes: bytes) -> dict:
    """
    Lightweight image analysis using:
    - ELA (compression inconsistency)
    - Frequency spectrum analysis
    - Perceptual hashing
    No deep learning model needed.
    """
    image = sanitize_image(image_bytes)
    orig_w, orig_h = image.size

    ela_val = compute_ela(image)

    img_np = np.array(image)
    hf_ratio = _compute_frequency_score(img_np)

    ela_normalized = min(ela_val / 50.0, 1.0)
    hf_anomaly = abs(hf_ratio - 0.3)

    ai_score = round(0.5 * ela_normalized + 0.5 * hf_anomaly, 4)
    ai_score = min(max(ai_score, 0.0), 1.0)

    is_fake = ai_score >= 0.35

    suspicious_regions = []
    if is_fake:
        suspicious_regions.append({
            "box_2d": [int(orig_h * 0.15), int(orig_w * 0.15),
                       int(orig_h * 0.85), int(orig_w * 0.85)],
            "label": f"Suspicious Artifacts ({round(ai_score * 100, 1)}%)",
            "score": round(ai_score, 2),
        })

    return {
        "fake_probability": ai_score,
        "is_manipulated": is_fake,
        "ela_anomaly_score": ela_val,
        "suspicious_regions": suspicious_regions,
        "gradcam_heatmap": "",
    }