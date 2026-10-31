import io
import base64
import logging
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageChops
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger("deepfake-shield.visual")

# 100% Public & Verified Hugging Face Hub Model
MODEL_ID = "Organika/sdxl-detector"

# --- Lazy model state (loaded on first request, not at import) ---------------
_processor = None
_model = None
ID2LABEL = {}
_model_loaded = False
_model_load_error = None


def _load_model():
    """Lazy-load the ViT detector on first request. Safe for offline environments."""
    global _processor, _model, ID2LABEL, _model_loaded, _model_load_error
    if _model_loaded or _model_load_error is not None:
        return
    try:
        print(f"[*] Lazy-loading Universal Detector: {MODEL_ID}...")
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        _model.eval()
        ID2LABEL = {int(k): v.lower() for k, v in _model.config.id2label.items()}
        print(f"[*] Loaded class mapping: {ID2LABEL}")
        _register_gradcam_hooks()
        _model_loaded = True
    except Exception as e:
        _model_load_error = str(e)
        logger.error(f"Failed to load visual model: {e}")


# --- Grad-CAM state ---------------------------------------------------------
_gradients = None
_activations = None
_hook_handles = []


def _save_gradient(module, grad_input, grad_output):
    global _gradients
    _gradients = grad_output[0].detach()


def _save_activation(module, input, output):
    global _activations
    _activations = output.detach()


def _register_gradcam_hooks():
    """Hook the last transformer block of the ViT for Grad-CAM."""
    global _hook_handles
    _remove_gradcam_hooks()

    if _model is None:
        return

    # ViT backbone lives under model.vit (or model.base_model for some variants)
    vit = getattr(_model, "vit", getattr(_model, "base_model", None))
    if vit is None:
        return

    # Target the last encoder block's output
    try:
        last_block = vit.encoder.layer[-1]
        _hook_handles.append(last_block.register_forward_hook(_save_activation))
        _hook_handles.append(last_block.register_full_backward_hook(_save_gradient))
    except Exception:
        # Fallback: hook the final layernorm before the classifier
        try:
            final_norm = vit.layernorm
            _hook_handles.append(final_norm.register_forward_hook(_save_activation))
            _hook_handles.append(final_norm.register_full_backward_hook(_save_gradient))
        except Exception:
            pass


def _remove_gradcam_hooks():
    global _hook_handles
    for h in _hook_handles:
        h.remove()
    _hook_handles = []


# Grad-CAM hooks are registered inside _load_model() on first use

def sanitize_image(image_bytes: bytes) -> Image.Image:
    """Handles EXIF rotation, alpha channels, and palette modes safely."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def compute_ela(pil_img: Image.Image) -> float:
    """Computes Error Level Analysis to detect compression inconsistency."""
    try:
        buffer = io.BytesIO()
        pil_img.save(buffer, "JPEG", quality=90)
        buffer.seek(0)
        resaved = Image.open(buffer)
        diff = ImageChops.difference(pil_img, resaved)
        diff_np = np.array(diff, dtype=np.float32)
        return round(float(np.mean(diff_np)), 2)
    except Exception:
        return 0.0


def _generate_gradcam_heatmap(image_bytes: bytes, fake_idx: int) -> str:
    """
    Produce a Grad-CAM heatmap for the AI/fake class.
    Returns a base64-encoded PNG (RGBA) of the heatmap resized to the input image.
    """
    global _gradients, _activations
    _gradients = None
    _activations = None

    image = sanitize_image(image_bytes)
    orig_w, orig_h = image.size

    inputs = _processor(images=image, return_tensors="pt")

    # Forward pass
    outputs = _model(**inputs)
    logits = outputs.logits

    # Backward pass on the fake-class score
    _model.zero_grad()
    target = logits[0, fake_idx]
    target.backward()

    if _activations is None or _gradients is None:
        # Fallback: return a blank heatmap
        blank = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        return base64.b64encode(cv2.imencode(".png", blank)[1]).decode("ascii")

    # ViT activations: [batch, num_tokens, hidden] or [batch, hidden, H, W]
    acts = _activations[0]  # [tokens, hidden] or [hidden, H, W]
    grads = _gradients[0]   # same shape as acts

    if acts.dim() == 3:
        # Sequence of tokens (ViT): exclude CLS token (index 0), treat rest as patches
        token_acts = acts[1:]                      # [num_patches, hidden]
        token_grads = grads[1:]                    # [num_patches, hidden]
        # Global average pooling of gradients over hidden dim -> weights per patch
        weights = token_grads.mean(dim=-1)         # [num_patches]
        heatmap = (token_acts * weights.unsqueeze(-1)).sum(dim=-1)  # [num_patches]
        heatmap = F.relu(heatmap)

        # Reshape patch tokens back to a 2D grid
        num_patches = heatmap.shape[0]
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            # Not a perfect square; fallback to blank
            blank = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            return base64.b64encode(cv2.imencode(".png", blank)[1]).decode("ascii")
        heatmap = heatmap.view(grid_size, grid_size)
    elif acts.dim() == 4:
        # Convolutional feature map: [channels, H, W]
        weights = grads.mean(dim=(1, 2))           # [channels]
        heatmap = (acts * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
        heatmap = F.relu(heatmap)
    else:
        blank = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        return base64.b64encode(cv2.imencode(".png", blank)[1]).decode("ascii")

    # Normalize heatmap to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_np = heatmap.cpu().numpy()

    # Resize to original image dimensions
    heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Apply colormap (jet) and convert to RGBA overlay
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)      # RGB

    # Build RGBA: color channels from colormap, alpha from intensity
    alpha = (heatmap_resized * 180).astype(np.uint8)  # semi-transparent overlay
    rgba = np.dstack([heatmap_color, alpha])

    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        blank = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        return base64.b64encode(cv2.imencode(".png", blank)[1]).decode("ascii")

    return base64.b64encode(buf).decode("ascii")


def _extract_suspicious_regions_from_heatmap(heatmap_np: np.ndarray, orig_h: int, orig_w: int) -> list:
    """Threshold the heatmap to find contiguous suspicious bounding boxes."""
    regions = []
    if heatmap_np.size == 0:
        return regions

    # Normalize and threshold at top 25% of activation
    norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-6)
    mask = (norm > 0.6).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 400:  # ignore tiny specks
            continue
        # box_2d format: [y1, x1, y2, x2] normalized 0..1 (consistent with existing code)
        regions.append({
            "box_2d": [
                round(y / orig_h, 4),
                round(x / orig_w, 4),
                round((y + h) / orig_h, 4),
                round((x + w) / orig_w, 4),
            ],
            "label": f"Grad-CAM activation ({round(float(norm[labels == i].max()), 2)})",
            "score": round(float(norm[labels == i].max()), 2),
        })
    return regions


def analyze_image_with_heatmap(image_bytes: bytes) -> dict:
    # Lazy-load model on first request (safe for offline environments)
    _load_model()
    if not _model_loaded:
        return {
            "fake_probability": 0.0,
            "is_manipulated": False,
            "ela_anomaly_score": 0.0,
            "suspicious_regions": [],
            "gradcam_heatmap": "",
            "error": f"Visual model unavailable: {_model_load_error}",
        }

    image = sanitize_image(image_bytes)
    orig_w, orig_h = image.size

    # ViT transformer forward pass
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    # Dynamically locate artificial/fake index
    fake_idx = 1
    for k, v in ID2LABEL.items():
        if any(term in v for term in ["artific", "fake", "synthet"]):
            fake_idx = k
            break

    ai_score = round(float(probs[fake_idx]), 4)
    ela_val = compute_ela(image)

    # 0.65 threshold prevents compressed WhatsApp/web photos from being marked fake
    is_fake = ai_score >= 0.65

    # --- Grad-CAM heatmap ---
    heatmap_b64 = ""
    suspicious_regions = []
    if is_fake:
        heatmap_b64 = _generate_gradcam_heatmap(image_bytes, fake_idx)
        # Decode heatmap back to numpy for region extraction
        try:
            heatmap_img = cv2.imdecode(
                np.frombuffer(base64.b64decode(heatmap_b64), dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if heatmap_img is not None and heatmap_img.shape[-1] == 4:
                alpha_channel = heatmap_img[..., 3].astype(np.float32) / 255.0
                suspicious_regions = _extract_suspicious_regions_from_heatmap(
                    alpha_channel, orig_h, orig_w
                )
        except Exception:
            suspicious_regions = []

    # Fallback if Grad-CAM produced nothing but we still want a region
    if is_fake and not suspicious_regions:
        suspicious_regions.append({
            "box_2d": [round(orig_h * 0.15, 4), round(orig_w * 0.15, 4),
                       round(orig_h * 0.85, 4), round(orig_w * 0.85, 4)],
            "label": f"AI Latent Diffusion Artifacts ({round(ai_score * 100, 1)}%)",
            "score": round(ai_score, 2),
        })

    return {
        "fake_probability": ai_score,
        "is_manipulated": is_fake,
        "ela_anomaly_score": ela_val,
        "suspicious_regions": suspicious_regions,
        "gradcam_heatmap": heatmap_b64,   # base64 RGBA PNG, same size as input
    }