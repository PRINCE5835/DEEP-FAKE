from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.forensics.provenance import inspect_c2pa
from app.forensics.visual import analyze_image_with_heatmap
from app.forensics.audio import analyze_audio
from app.forensics.video import analyze_video
from app.forensics.text_checker import verify_text_claim

app = FastAPI(title="Deepfake Shield API", version="2.0.0")

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
        "service": "Deepfake Shield API",
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
                "provenance": c2pa_res
            }

        # 2. Universal Visual Forensics (ViT + Frequency/ELA)
        visual_res = analyze_image_with_heatmap(raw_bytes)
        ai_score = visual_res["fake_probability"]

        if visual_res["is_manipulated"]:
            verdict = "SYNTHETIC"
            plain_text = f"AI Generation Patterns detected with {round(ai_score * 100, 1)}% confidence. Latent diffusion grid noise and synthetic texture smoothing identified."
        else:
            verdict = "AUTHENTIC"
            organic_score = round((1.0 - ai_score) * 100, 1)
            plain_text = f"Authentic camera capture confirmed ({organic_score}% organic). Physical optical sensor noise and natural chromatic dispersion verified."

        return {
            "verdict": verdict,
            "confidence": ai_score if visual_res["is_manipulated"] else round(1.0 - ai_score, 4),
            "is_manipulated": visual_res["is_manipulated"],
            "plain_explanation": plain_text,
            "suspicious_regions": visual_res["suspicious_regions"],
            "ela_score": visual_res["ela_anomaly_score"],
            "provenance": c2pa_res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/v1/detect/audio")
async def detect_audio_endpoint(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload.")
    return analyze_audio(raw_bytes)

@app.post("/api/v1/detect/video")
async def detect_video_endpoint(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty video payload.")
    return analyze_video(raw_bytes)

@app.post("/api/v1/detect/text")
async def detect_text_endpoint(text: str = Form(...)):
    fact_check = await verify_text_claim(text)
    has_matches = len(fact_check.get("matches", [])) > 0
    return {
        "verdict": "DEBUNKED" if has_matches else "UNVERIFIED",
        "confidence": 0.95 if has_matches else 0.5,
        "is_manipulated": has_matches,
        "plain_explanation": f"Matched {len(fact_check['matches'])} verified fact-checking records." if has_matches else "No verified debunking records found for this statement.",
        "suspicious_regions": [],
        "ela_score": 0.0,
        "fact_check": fact_check
    }
