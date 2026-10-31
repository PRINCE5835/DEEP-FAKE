# 🛡️ DEEP-SHIELD — Multimodal AI Forensic Defense Suite

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Chrome MV3](https://img.shields.io/badge/Chrome_Extension-Manifest_V3-4285F4.svg?logo=googlechrome&logoColor=white)](https://developer.chrome.com/docs/extensions/mv3/)
[![C2PA](https://img.shields.io/badge/Standard-C2PA_%2F_CAI-blue.svg)](https://c2pa.org/)
[![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch_%26_Wav2Vec_2.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

**DEEP-SHIELD** is a multimodal AI forensic system and browser extension designed to protect internet users from synthetic media deception in real-time. It detects deepfake images, cloned audio, AI-generated disinformation, and extracts cryptographically signed C2PA provenance manifests directly *in-situ* on any webpage (Twitter/X, Facebook, YouTube, Reddit, news portals).

---

## 🏛️ System Architecture

```
deepfake-shield/
├── backend/
│   ├── app/
│   │   ├── forensics/
│   │   │   ├── __init__.py
│   │   │   ├── provenance.py      # C2PA JUMBF extraction, XMP, & AI generation markers
│   │   │   ├── visual.py          # Vision model, Grad-CAM saliency heatmaps, & ELA
│   │   │   ├── audio.py           # Wav2Vec 2.0 & neural vocoder voice cloning detector
│   │   │   └── text_checker.py    # FactCheck ClaimReview API & stylometric AI detection
│   │   ├── __init__.py
│   │   └── main.py                # High-performance FastAPI routes & async orchestrator
│   ├── requirements.txt           # Python dependencies
│   └── .env                       # Environment configuration
└── extension/
    ├── manifest.json              # Chrome Manifest V3 configuration
    ├── background.js              # Service worker & right-click context menu dispatcher
    ├── content.js                 # In-situ DOM HUD overlay & canvas pixel capture
    ├── popup.html                 # Direct upload modal & quick inspector
    ├── popup.css                  # Cyber-glassmorphism theme styles
    ├── popup.js                   # Popup upload controller & real-time visualizer
    └── icons/                     # Extension icons (16px, 48px, 128px)
```

---

## 🔬 Core Forensic Engines

### 1. 🔍 C2PA Provenance & Cryptographic Manifests (`provenance.py`)
- **Binary JUMBF Scanner**: Parses JPEG APP11 markers, PNG chunks, and MP4/MOV UUID boxes for C2PA (`urn:c2pa`) claims.
- **Assertion Verification**: Validates digital source type (`trainedAlgorithmicMedia` vs `photographicImage`), signer identity, and certificate validity.
- **AI Generator Fingerprints**: Extracts generation prompts, sampler parameters, and signatures from Midjourney, DALL·E 3, Adobe Firefly, and Stable Diffusion.

### 2. 👁️ Visual Forensics, Grad-CAM & ELA (`visual.py`)
- **Error Level Analysis (ELA)**: Re-compresses images at fixed quality factors to reveal localized compression rate variance, detecting digital splices and inpainting.
- **2D FFT Frequency Analysis**: Computes the radial high-frequency power decay in the Fourier spectrum to detect characteristic checkerboard artifacts and synthesis anomalies.
- **Vision Model & Grad-CAM**: Uses deep convolutional neural activations to produce Class Activation Maps (Grad-CAM), highlighting manipulated facial regions (eyes, mouth, boundary seams) as colorized heatmaps.

### 3. 🎙️ Wav2Vec 2.0 Voice Cloning Detector (`audio.py`)
- **Neural Vocoder Brick-Wall Cutoff**: Identifies artificial frequency cutoffs at 16 kHz or 24 kHz typical of TTS neural vocoders (HiFi-GAN, ElevenLabs, VITS).
- **Pitch (F0) Micro-Jitter**: Measures fundamental frequency micro-perturbations via autocorrelation. Synthetic voices exhibit unnaturally flat micro-tremors or erratic jump discontinuities.
- **Mel-Spectrogram Generation**: Produces dynamic spectral visualizations identifying unnatural phase silence and noise gating.

### 4. 📝 Stylometrics & Claim Fact-Checking (`text_checker.py`)
- **Forensic Stylometrics**: Evaluates sentence burstiness ($\sigma/\mu$), lexical diversity (Type-Token Ratio), and flags overused LLM transitions.
- **FactCheck ClaimReview Integration**: Cross-references viral claims against the Google Fact Check Tools API and an offline curated database of debunked deepfake incidents.

---

## 🚀 Quickstart Guide

### 1. Start the Backend API

```powershell
# 1. Navigate to backend directory
cd deepfake-shield/backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Status: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 2. Install the Chrome Extension

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle switch in the top-right corner).
3. Click **Load unpacked**.
4. Select the `deepfake-shield/extension` folder.
5. The 🛡️ **DEEP-SHIELD** extension is now active!

---

## 💡 How to Use

### Method A: Right-Click In-Situ Inspection (Browsing the Web)
1. Right-click on any image, video, audio clip, or highlighted text on any webpage.
2. Select **🛡️ DEEP-SHIELD** from the context menu:
   - *🔍 Scan Image (Grad-CAM & ELA Heatmap)*
   - *🎙️ Scan Voice Clone / Vocoder Artifacts*
   - *📝 Verify Claim & AI Stylometrics*
3. The glassmorphic **Forensic HUD** instantly docks onto the media, showing the **Shield Trust Score (0-100%)**, visual heatmaps, and C2PA manifest provenance.

### Method B: Standalone Extension Popup
1. Click the **🛡️ DEEP-SHIELD** icon in your browser toolbar.
2. Drag and drop any image/audio file or paste a URL/claim.
3. Click **Analyze** to generate an immediate audit report with interactive ELA and Grad-CAM visualizations.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Hardware acceleration, CUDA status & module health |
| `POST` | `/api/analyze/image` | Image forensics (Grad-CAM + ELA + C2PA provenance) |
| `POST` | `/api/analyze/audio` | Voice clone detection, vocoder cutoffs & spectrogram |
| `POST` | `/api/analyze/text` | AI stylometrics & FactCheck claim verification |
| `POST` | `/api/analyze/media` | Unified multimodal orchestrator with automatic type detection |

---

## 🛡️ Hackathon Highlights

- **Full Multimodal Coverage**: Images, Audio/Voice, Text, and Cryptographic Provenance.
- **Explainable AI (XAI)**: Generates Grad-CAM activation maps and ELA difference heatmaps so users can see *why* an image is fake, not just a black-box probability.
- **C2PA Standard Alignment**: Bridges deep learning with official Coalition for Content Provenance and Authenticity standards.
- **Graceful Fallbacks**: Operates seamlessly across both high-end GPU workstations and lightweight CPU environments.

