"""
Reverse-image search via perceptual hashing.

Compares an uploaded image's perceptual hash against a local registry of
known AI-generated / debunked images. Returns match confidence and source
metadata so users can cross-check against trusted sources.

Optional external lookup (SerpAPI / Google Lens) can be enabled via
REVERSE_IMAGE_SEARCH_API_KEY in backend/.env.
"""

import io
import json
import os
from typing import Dict, Any, List, Optional
from PIL import Image
import imagehash

# --- Local registry of known fakes -------------------------------------------
# Each entry: { "hash": "<hex phash>", "source": "...", "label": "AI-generated" }
# Populate this with known deepfake image hashes for your use case.
KNOWN_FAKES_DB: List[Dict[str, Any]] = [
    # Example placeholders — replace with real known-fake hashes:
    # {
    #   "hash": "a1b2c3d4e5f67890",
    #   "source": "Known deepfake repository",
    #   "label": "AI-generated face swap",
    # },
]

# Threshold: hamming distance <= 5 is considered a match (64-bit phash)
HASH_MATCH_THRESHOLD = 5

# --- Optional external API ---------------------------------------------------
REVERSE_SEARCH_API_KEY = os.getenv("REVERSE_IMAGE_SEARCH_API_KEY", "")
REVERSE_SEARCH_API_URL = os.getenv(
    "REVERSE_IMAGE_SEARCH_API_URL",
    "https://api.serpapi.com/search",  # placeholder; configure per provider
)


def compute_phash(image_bytes: bytes) -> str:
    """Compute a 64-bit perceptual hash (hex string) for an image."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    phash = imagehash.phash(img, hash_size=16)
    return str(phash)


def _hamming_distance_hex(h1: str, h2: str) -> int:
    """Hamming distance between two hex hash strings."""
    if len(h1) != len(h2):
        return 999
    n1 = int(h1, 16)
    n2 = int(h2, 16)
    return bin(n1 ^ n2).count("1")


def search_local_registry(image_hash: str) -> List[Dict[str, Any]]:
    """Search the local known-fakes registry for similar hashes."""
    matches: List[Dict[str, Any]] = []
    for entry in KNOWN_FAKES_DB:
        dist = _hamming_distance_hex(image_hash, entry["hash"])
        if dist <= HASH_MATCH_THRESHOLD:
            confidence = round(1.0 - (dist / 64.0), 4)
            matches.append({
                "source": entry.get("source", "Local registry"),
                "label": entry.get("label", "Potential match"),
                "confidence": confidence,
                "hamming_distance": dist,
                "match_type": "local_perceptual_hash",
            })
    return matches


async def search_external_api(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Optional external reverse-image search (e.g. SerpAPI / Google Lens).
    Returns empty list if no API key is configured.
    """
    if not REVERSE_SEARCH_API_KEY:
        return []

    try:
        import httpx
        # Buffer image as multipart for external reverse-image endpoint
        files = {"image": ("query.jpg", image_bytes, "image/jpeg")}
        params = {
            "api_key": REVERSE_SEARCH_API_KEY,
            "engine": "google_lens",  # or "google_reverse_image"
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(REVERSE_SEARCH_API_URL, files=files, params=params)
            if resp.status_code != 200:
                return []
            data = resp.json()

        matches: List[Dict[str, Any]] = []
        # SerpAPI Google Lens results structure (adjust per provider)
        for item in data.get("visual_matches", [])[:5]:
            matches.append({
                "source": item.get("source", "External reverse-image search"),
                "label": item.get("title", "Visual match"),
                "confidence": round(float(item.get("confidence", 0.5)), 4),
                "url": item.get("link", "#"),
                "match_type": "external_api",
            })
        return matches
    except Exception:
        return []


async def reverse_image_search(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run reverse-image search: local registry first, then external API.
    Returns a dict with match results and overall assessment.
    """
    image_hash = compute_phash(image_bytes)

    local_matches = search_local_registry(image_hash)

    # Run optional external API lookup (async)
    external_matches = await search_external_api(image_bytes)

    all_matches = list(local_matches) + list(external_matches)
    best_confidence = max([m["confidence"] for m in all_matches], default=0.0)

    result: Dict[str, Any] = {
        "image_hash": image_hash,
        "local_matches": local_matches,
        "external_matches": external_matches,
        "all_matches": all_matches,
        "is_known_fake": len(all_matches) > 0,
        "best_confidence": best_confidence,
    }
    return result


# Backwards-compatible alias
search_image_hash = reverse_image_search
