import io
import os
import tempfile
import logging
import cv2
import numpy as np
from PIL import Image
from app.forensics.visual import analyze_image_with_heatmap

logger = logging.getLogger("deepfake-shield.video")

# Safely initialize Haar Cascade if present; otherwise fall back gracefully
face_cascade = None
eye_cascade = None
try:
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        cascade_dir = cv2.data.haarcascades
        face_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
        eye_path = os.path.join(cascade_dir, 'haarcascade_eye.xml')
        if os.path.exists(face_path) and hasattr(cv2, 'CascadeClassifier'):
            face_cascade = cv2.CascadeClassifier(face_path)
        if os.path.exists(eye_path) and hasattr(cv2, 'CascadeClassifier'):
            eye_cascade = cv2.CascadeClassifier(eye_path)
except Exception:
    face_cascade = None
    eye_cascade = None


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


def _eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """Compute eye aspect ratio from 6 eye landmarks (approximation from Haar eyes)."""
    if eye_landmarks is None or len(eye_landmarks) < 6:
        return 0.0
    # Use simple vertical/horizontal ratio approximation
    try:
        vertical = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        if horizontal == 0:
            return 0.0
        return vertical / horizontal
    except Exception:
        return 0.0


def _detect_blinks(gray_roi: np.ndarray) -> float:
    """Detect eye aspect ratio in a face region; returns average EAR (lower = eyes more closed)."""
    if eye_cascade is None or eye_cascade.empty():
        return 0.5  # neutral
    try:
        eyes = eye_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(eyes) < 2:
            return 0.5
        ears = []
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_roi = gray_roi[ey:ey + eh, ex:ex + ew]
            # Approximate EAR from eye region aspect ratio
            ear = ew / (eh + 1e-6)
            ears.append(min(ear, 1.0))
        return float(np.mean(ears))
    except Exception:
        return 0.5


def _optical_flow_consistency(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Compute mean optical flow magnitude between two frames.
    Unusually high or inconsistent flow can indicate morphing/warping artifacts.
    """
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))
    except Exception:
        return 0.0


def analyze_video(video_bytes: bytes, sample_fps: int = 1) -> dict:
    """Samples frames over time and calculates deepfake probability score with temporal consistency."""
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
    ear_values = []
    flow_magnitudes = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            analyzed_frames += 1
            timestamp_sec = round(frame_count / video_fps, 2)

            target_crop = extract_primary_region(frame)
            gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)

            # Blink / eye aspect ratio
            ear = _detect_blinks(gray)
            ear_values.append(ear)

            # Optical flow vs previous frame
            if prev_gray is not None:
                flow_mag = _optical_flow_consistency(prev_gray, gray)
                flow_magnitudes.append(flow_mag)
            prev_gray = gray

            # Per-frame image forensics
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
            "ela_score": 0.0,
            "temporal_consistent": False,
        }

    avg_fake_score = sum(fake_scores) / len(fake_scores)
    max_fake_score = max(fake_scores)

    # --- Temporal consistency signals ---
    # Blink analysis: humans blink regularly; absence or unnatural pattern = suspicious
    blink_anomaly = 0.0
    if ear_values:
        ear_arr = np.array(ear_values)
        # Low EAR values indicate eyes closed; count frames with very low EAR (blinks)
        blink_frames = int(np.sum(ear_arr < 0.25))
        blink_rate = blink_frames / len(ear_arr)
        # Natural blink rate is roughly 0.02-0.05 per frame at 1fps sampling
        if blink_rate < 0.005 or blink_rate > 0.3:
            blink_anomaly = min(abs(blink_rate - 0.03) * 3, 1.0)

    # Optical flow consistency: sudden erratic flow = morphing artifacts
    flow_anomaly = 0.0
    if flow_magnitudes:
        flow_arr = np.array(flow_magnitudes)
        flow_std = float(np.std(flow_arr))
        flow_mean = float(np.mean(flow_arr))
        # High variance in flow = inconsistent motion
        if flow_mean > 0:
            flow_anomaly = min(flow_std / (flow_mean + 1e-6), 1.0)

    # --- Combined verdict ---
    temporal_anomaly_score = round((blink_anomaly + flow_anomaly) / 2.0, 4)
    combined_fake_score = round(0.6 * avg_fake_score + 0.4 * temporal_anomaly_score, 4)
    max_fake_score = max(max_fake_score, combined_fake_score)

    is_tampered = (avg_fake_score > 0.45 or max_fake_score > 0.75 or temporal_anomaly_score > 0.6)

    if is_tampered:
        verdict = "SYNTHETIC"
        explanation = (
            f"Video indicates temporal facial tampering. "
            f"Avg frame manipulation: {round(avg_fake_score * 100, 1)}%, "
            f"temporal anomaly: {round(temporal_anomaly_score * 100, 1)}%. "
            f"Suspicious timestamps: {suspicious_timestamps[:5]}s."
        )
    else:
        verdict = "AUTHENTIC"
        explanation = (
            f"Analyzed {analyzed_frames} sampled frames. "
            f"Temporal consistency and facial noise balance confirm authenticity. "
            f"Blink anomaly: {round(blink_anomaly * 100, 1)}%, flow anomaly: {round(flow_anomaly * 100, 1)}%."
        )

    return {
        "verdict": verdict,
        "confidence": round(combined_fake_score if is_tampered else (1.0 - combined_fake_score), 4),
        "is_manipulated": is_tampered,
        "plain_explanation": explanation,
        "suspicious_regions": [],
        "ela_score": 0.0,
        "analyzed_frames": analyzed_frames,
        "suspicious_timestamps": suspicious_timestamps[:5],
        "avg_frame_fake_score": round(avg_fake_score, 4),
        "temporal_anomaly_score": temporal_anomaly_score,
        "blink_anomaly": round(blink_anomaly, 4),
        "flow_anomaly": round(flow_anomaly, 4),
        "temporal_consistent": not is_tampered,
    }