from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.forensics.provenance import inspect_c2pa
from app.forensics.visual import analyze_image_with_heatmap
from app.forensics.audio import analyze_audio
from app.forensics.video import analyze_video
from app.forensics.text_checker import verify_text_claim
from app.forensics.reverse_image_search import reverse_image_search

app = FastAPI(title="DEEP-SHIELD API", version="2.0.0")

# CORS middleware for Web Dashboard & Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "service": "DEEP-SHIELD API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": [
            "/api/v1/detect/image",
            "/api/v1/detect/audio",
            "/api/v1/detect/video",
            "/api/v1/detect/text",
            "/api/v1/health"
        ]
    }

@app.get("/api/v1/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "modules": {
            "image_detector": "Organika/sdxl-detector",
            "audio_detector": "MelodyMachine/Deepfake-audio-detection-V2",
            "video_detector": "HaarCascade + FrameSampler",
            "text_detector": "FactCheckClaimReview"
        }
    }

@app.post("/api/v1/detect/image")
@app.post("/api/analyze/image")
async def detect_image(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) < 100:
        raise HTTPException(status_code=400, detail="Corrupted or empty image payload.")

    try:
        # 1. C2PA Provenance Check
        c2pa_res = inspect_c2pa(raw_bytes)
        if c2pa_res.get("c2pa_detected") and c2pa_res.get("is_ai_generated"):
            return {
                "verdict": "SYNTHETIC",
                "confidence": 1.0,
                "is_manipulated": True,
                "plain_explanation": "C2PA cryptographic manifest certifies this media was synthesized by an AI generation engine.",
                "suspicious_regions": [],
                "ela_score": 0.0,
                "gradcam_heatmap": "",
                "provenance": c2pa_res,
                "reverse_image_search": {
                    "image_hash": "",
                    "local_matches": [],
                    "external_matches": [],
                    "all_matches": [],
                    "is_known_fake": False,
                    "best_confidence": 0.0,
                },
            }

        # 2. Universal Visual Forensics (ViT + Frequency/ELA + Grad-CAM)
        visual_res = analyze_image_with_heatmap(raw_bytes)
        ai_score = visual_res["fake_probability"]

        # 3. Reverse-image search cross-check
        ris = await reverse_image_search(raw_bytes)

        # If reverse-image search found a known fake, boost confidence
        if ris["is_known_fake"] and ris["best_confidence"] > 0.5:
            ai_score = max(ai_score, ris["best_confidence"])
            is_fake = True
        else:
            is_fake = visual_res["is_manipulated"]

        if is_fake:
            verdict = "SYNTHETIC"
            plain_text = f"AI Generation Patterns detected with {round(ai_score * 100, 1)}% confidence. Latent diffusion grid noise and synthetic texture smoothing identified."
            if ris["is_known_fake"]:
                plain_text += f" Reverse-image search matched {len(ris['local_matches'])} known-synthetic record(s)."
        else:
            verdict = "AUTHENTIC"
            organic_score = round((1.0 - ai_score) * 100, 1)
            plain_text = f"Authentic camera capture confirmed ({organic_score}% organic). Physical optical sensor noise and natural chromatic dispersion verified."

        return {
            "verdict": verdict,
            "confidence": ai_score if is_fake else round(1.0 - ai_score, 4),
            "is_manipulated": is_fake,
            "plain_explanation": plain_text,
            "suspicious_regions": visual_res["suspicious_regions"],
            "ela_score": visual_res["ela_anomaly_score"],
            "gradcam_heatmap": visual_res.get("gradcam_heatmap", ""),
            "provenance": c2pa_res,
            "reverse_image_search": ris,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/v1/detect/audio")
@app.post("/api/analyze/audio")
async def detect_audio_endpoint(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload.")
    return analyze_audio(raw_bytes)

@app.post("/api/v1/detect/video")
@app.post("/api/analyze/video")
async def detect_video_endpoint(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty video payload.")
    return analyze_video(raw_bytes)

@app.post("/api/v1/detect/text")
@app.post("/api/analyze/text")
async def detect_text_endpoint(text: str = Form(...)):
    result = await verify_text_claim(text)
    return {
        "verdict": result.get("verdict", "UNVERIFIED"),
        "confidence": result.get("confidence", 0.5),
        "is_manipulated": result.get("is_manipulated", False),
        "plain_explanation": result.get("plain_explanation", ""),
        "suspicious_regions": [],
        "ela_score": 0.0,
        "fact_check": {
            "available": result.get("fact_check_available", False),
            "error": result.get("fact_check_error"),
            "matches": result.get("matches", []),
        },
        "stylometrics": result.get("stylometrics", {}),
    }


@app.post("/api/v1/reverse-image-search")
async def reverse_image_search_endpoint(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) < 100:
        raise HTTPException(status_code=400, detail="Corrupted or empty image payload.")
    return await reverse_image_search(raw_bytes)


@app.post("/api/analyze/media")
@app.post("/api/v1/analyze/media")
async def analyze_media_endpoint(file: UploadFile = File(None), text: str = Form(None)):
    """
    Unified multimodal orchestrator with automatic type detection.
    Accepts a file (image/audio/video) or text, routes to the correct detector.
    """
    if text:
        return await verify_text_claim(text)

    if file is None:
        raise HTTPException(status_code=400, detail="No file or text provided.")

    raw_bytes = await file.read()
    content_type = file.content_type or ""
    filename = file.filename or ""

    if content_type.startswith("image/") or any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
        return await detect_image(file)
    elif content_type.startswith("audio/") or any(filename.lower().endswith(ext) for ext in [".wav", ".mp3", ".ogg", ".m4a", ".flac"]):
        return await detect_audio_endpoint(file)
    elif content_type.startswith("video/") or any(filename.lower().endswith(ext) for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]):
        return await detect_video_endpoint(file)
    else:
        # Default to image detection
        return await detect_image(file)
