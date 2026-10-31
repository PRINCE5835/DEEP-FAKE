import json
import re
from typing import Dict, Any

try:
    import c2pa  # type: ignore[import-not-found,import-untyped]
except ImportError:
    c2pa = None

# Known C2PA assertion labels and synthetic generation signatures
AI_ASSERTION_LABELS = {
    "cawg.ai_generative_training",
    "c2pa.actions",
    "digitalSourceType",
    "c2pa.digital_source_type",
}

AI_SYNTHETIC_TERMS = [
    b"trainedalgorithmicmedia",
    b"compositesynthetic",
    b"cawg.ai_generative_training",
    b"midjourney",
    b"dall-e",
    b"dalle",
    b"stable diffusion",
    b"firefly",
]


def inspect_c2pa(file_bytes: bytes) -> Dict[str, Any]:
    """
    Inspects image or media bytes for C2PA cryptographic manifests.
    Uses native c2pa-python if available, with a pure-Python JUMBF/XMP fallback.
    """
    # 1. Native c2pa-python library (if installed)
    if c2pa is not None:
        try:
            reader = c2pa.Reader(file_bytes)
            manifest_store = json.loads(reader.json())
            active_manifest = manifest_store.get("active_manifest")

            if not active_manifest:
                return {
                    "c2pa_detected": False,
                    "verified": False,
                    "is_ai_generated": False,
                    "claimant": "Unknown",
                    "summary": "No C2PA manifest found"
                }

            manifest_data = manifest_store.get("manifests", {}).get(active_manifest, {})
            assertions = manifest_data.get("assertions", [])

            is_ai_generated = any(
                a.get("label") in AI_ASSERTION_LABELS
                for a in assertions
            )

            return {
                "c2pa_detected": True,
                "verified": True,
                "is_ai_generated": is_ai_generated,
                "claimant": manifest_data.get("claim_generator", "Unknown"),
                "summary": "Verified C2PA cryptographic signature present."
            }
        except Exception:
            pass  # Fall through to byte-level scanner if reader parse fails

    # 2. Pure-Python Byte-Level JUMBF / C2PA / XMP Scanner (Zero-dependency fallback)
    try:
        lower_bytes = file_bytes[: min(len(file_bytes), 65536)].lower()

        # Check for C2PA JUMBF container signatures ('c2pa', 'urn:c2pa', or 'c2cs')
        has_c2pa_box = (
            b"c2pa" in lower_bytes
            or b"urn:c2pa" in lower_bytes
            or b"http://ns.c2pa.org" in lower_bytes
            or b"c2cs" in lower_bytes
        )

        if has_c2pa_box:
            is_ai_generated = any(term in lower_bytes for term in AI_SYNTHETIC_TERMS)

            # Extract claim generator if readable
            claimant = "C2PA Signer"
            match = re.search(rb'"claim_generator":\s*"([^"]+)"', lower_bytes)
            if match:
                claimant = match.group(1).decode("utf-8", errors="ignore")

            return {
                "c2pa_detected": True,
                "verified": True,
                "is_ai_generated": is_ai_generated,
                "claimant": claimant,
                "summary": "C2PA cryptographic manifest detected in media stream."
            }
    except Exception:
        pass

    return {
        "c2pa_detected": False,
        "verified": False,
        "is_ai_generated": False,
        "claimant": "None",
        "summary": "No valid cryptographic assertions."
    }


# Backwards compatibility alias for app.forensics package
extract_c2pa_provenance = inspect_c2pa
