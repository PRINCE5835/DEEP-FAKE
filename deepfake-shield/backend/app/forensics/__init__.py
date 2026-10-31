"""
Forensic Analysis Subsystems:
- Provenance (C2PA / JUMBF / XMP metadata extraction)
- Visual (Deepfake vision model, Grad-CAM heatmap, Error Level Analysis)
- Audio (Wav2Vec 2.0 voice cloning and synthetic acoustic detection)
- Text Checker (FactCheck ClaimReview API)
"""

try:
    from .provenance import inspect_c2pa, extract_c2pa_provenance
except ImportError:
    inspect_c2pa = None  # type: ignore
    extract_c2pa_provenance = None  # type: ignore

try:
    from .visual import analyze_image_with_heatmap
    analyze_visual_forensics = analyze_image_with_heatmap
except ImportError:
    analyze_image_with_heatmap = None  # type: ignore
    analyze_visual_forensics = None  # type: ignore

try:
    from .audio import analyze_audio
    analyze_audio_forensics = analyze_audio
except ImportError:
    analyze_audio = None  # type: ignore
    analyze_audio_forensics = None  # type: ignore

try:
    from .video import analyze_video
    analyze_video_forensics = analyze_video
except ImportError:
    analyze_video = None  # type: ignore
    analyze_video_forensics = None  # type: ignore

try:
    from .text_checker import verify_text_claim
    analyze_text_forensics = verify_text_claim
except ImportError:
    verify_text_claim = None  # type: ignore
    analyze_text_forensics = None  # type: ignore

try:
    from .reverse_image_search import reverse_image_search
    analyze_reverse_image_search = reverse_image_search
except ImportError:
    reverse_image_search = None  # type: ignore
    analyze_reverse_image_search = None  # type: ignore

__all__ = [
    "inspect_c2pa",
    "extract_c2pa_provenance",
    "analyze_image_with_heatmap",
    "analyze_visual_forensics",
    "analyze_audio",
    "analyze_audio_forensics",
    "analyze_video",
    "analyze_video_forensics",
    "verify_text_claim",
    "analyze_text_forensics",
    "reverse_image_search",
    "analyze_reverse_image_search",
]
