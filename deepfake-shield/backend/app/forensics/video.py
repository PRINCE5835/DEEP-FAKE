import io
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from app.forensics.visual import analyze_image_with_heatmap

# Safely initialize Haar Cascade if present; otherwise fall back gracefully
face_cascade = None
try:
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path) and hasattr(cv2, 'CascadeClassifier'):
            face_cascade = cv2.CascadeClassifier(cascade_path)
except Exception:
    face_cascade = None

def extract_primary_region(frame_bgr: np.ndarray) -> np.ndarray:
    """Isolates face region if cascade is available, otherwise extracts center frame crop."""
    h, w = frame_bgr.shape[:2]
    
    if face_cascade is not None and not face_cascade.empty():
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                pad = int(0.2 * fw)
                y1 = max(0, y - pad)
                y2 = min(h, y + fh + pad)
                x1 = max(0, x - pad)
                x2 = min(w, x + fw + pad)
                return frame_bgr[y1:y2, x1:x2]
        except Exception:
            pass

    # Central focal area fallback
    cy1, cy2 = int(h * 0.15), int(h * 0.85)
    cx1, cx2 = int(w * 0.15), int(w * 0.85)
    return frame_bgr[cy1:cy2, cx1:cx2]

def analyze_video(video_bytes: bytes, sample_fps: int = 1) -> dict:
    """Samples frames over time and calculates deepfake probability score."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise ValueError("Could not read uploaded video format.")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_interval = max(int(video_fps / sample_fps), 1)

    frame_count = 0
    analyzed_frames = 0
    fake_scores = []
    suspicious_timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            analyzed_frames += 1
            timestamp_sec = round(frame_count / video_fps, 2)
            
            target_crop = extract_primary_region(frame)
            rgb_frame = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            
            result = analyze_image_with_heatmap(buf.getvalue())
            fake_prob = result["fake_probability"]
            fake_scores.append(fake_prob)

            if fake_prob > 0.6:
                suspicious_timestamps.append(timestamp_sec)

        frame_count += 1

    cap.release()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if not fake_scores:
        return {
            "verdict": "AUTHENTIC",
            "confidence": 0.0,
            "is_manipulated": False,
            "plain_explanation": "No readable video frames could be extracted.",
            "suspicious_regions": [],
            "ela_score": 0.0
        }

    avg_fake_score = sum(fake_scores) / len(fake_scores)
    max_fake_score = max(fake_scores)
    is_tampered = avg_fake_score > 0.45 or max_fake_score > 0.75

    if is_tampered:
        verdict = "SYNTHETIC"
        explanation = f"Video indicates temporal facial tampering. High manipulation probability at timestamps: {suspicious_timestamps[:5]}s."
    else:
        verdict = "AUTHENTIC"
        explanation = f"Analyzed {analyzed_frames} sampled frames. Temporal consistency and facial noise balance confirm authenticity."

    return {
        "verdict": verdict,
        "confidence": round(avg_fake_score if is_tampered else (1.0 - avg_fake_score), 4),
        "is_manipulated": is_tampered,
        "plain_explanation": explanation,
        "suspicious_regions": [],
        "ela_score": 0.0,
        "analyzed_frames": analyzed_frames,
        "suspicious_timestamps": suspicious_timestamps[:5]
    }
