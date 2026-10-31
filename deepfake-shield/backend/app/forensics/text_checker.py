import os
import re
import logging
import httpx
from dotenv import load_dotenv

logger = logging.getLogger("deepfake-shield.text")

# Load variables from .env
load_dotenv()

FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

# Common LLM transition / hedge phrases that appear frequently in AI-generated text
LLM_TRANSITION_PHRASES = [
    r"\bIt is important to note\b",
    r"\bIn conclusion\b",
    r"\bFurthermore\b",
    r"\bMoreover\b",
    r"\bAdditionally\b",
    r"\bConsequently\b",
    r"\bNevertheless\b",
    r"\bNonetheless\b",
    r"\bThat being said\b",
    r"\bOn the other hand\b",
    r"\bIn summary\b",
    r"\bTo summarize\b",
    r"\bIt is worth mentioning\b",
    r"\bNotably\b",
    r"\bSignificantly\b",
    r"\bCrucially\b",
    r"\bEvidently\b",
    r"\bClearly\b",
    r"\bObviously\b",
    r"\bUndoubtedly\b",
]

# Overconfident / absolute claim markers common in misinformation
ABSOLUTE_CLAIM_PHRASES = [
    r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bnobody\b",
    r"\ball\b", r"\bnone\b", r"\bcompletely\b", r"\btotally\b",
    r"\bdefinitely\b", r"\bcertainly\b", r"\babsolutely\b",
]


def _split_sentences(text: str) -> list:
    """Simple sentence splitter."""
    sentences = re.split(r"[.!?]+[\s\n]+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _compute_stylometrics(text: str) -> dict:
    """
    Compute forensic stylometric features:
    - burstiness: std/mean of sentence lengths (high = uneven, LLM-like)
    - ttr: type-token ratio (vocabulary diversity; low = repetitive, LLM-like)
    - transition_density: LLM transition phrases per 100 words
    - absolute_claim_density: absolute-claim phrases per 100 words
    """
    sentences = _split_sentences(text)
    words = re.findall(r"\b\w+\b", text.lower())
    num_words = len(words)
    num_sentences = len(sentences)

    if num_words == 0:
        return {
            "burstiness": 0.0,
            "ttr": 0.0,
            "transition_density": 0.0,
            "absolute_claim_density": 0.0,
            "num_words": 0,
            "num_sentences": 0,
        }

    # Burstiness: sigma / mu of sentence lengths
    sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    if sentence_lengths:
        mean_len = sum(sentence_lengths) / len(sentence_lengths)
        if mean_len > 0:
            variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            burstiness = (variance ** 0.5) / mean_len
        else:
            burstiness = 0.0
    else:
        burstiness = 0.0

    # Type-Token Ratio
    unique_words = len(set(words))
    ttr = unique_words / num_words

    # Transition phrase density (per 100 words)
    transition_count = 0
    for phrase in LLM_TRANSITION_PHRASES:
        transition_count += len(re.findall(phrase, text, flags=re.IGNORECASE))

    absolute_count = 0
    for phrase in ABSOLUTE_CLAIM_PHRASES:
        absolute_count += len(re.findall(phrase, text, flags=re.IGNORECASE))

    transition_density = (transition_count / num_words) * 100 if num_words > 0 else 0.0
    absolute_claim_density = (absolute_count / num_words) * 100 if num_words > 0 else 0.0

    # Composite AI-likelihood score (0..1)
    # High burstiness, low TTR, high transition density => more AI-like
    ai_likelihood = 0.0
    if burstiness > 0.5:
        ai_likelihood += 0.3
    if ttr < 0.5:
        ai_likelihood += 0.3
    if transition_density > 5.0:
        ai_likelihood += 0.2
    if absolute_claim_density > 3.0:
        ai_likelihood += 0.2
    ai_likelihood = min(ai_likelihood, 1.0)

    return {
        "burstiness": round(burstiness, 4),
        "ttr": round(ttr, 4),
        "transition_density": round(transition_density, 4),
        "absolute_claim_density": round(absolute_claim_density, 4),
        "num_words": num_words,
        "num_sentences": num_sentences,
        "stylometric_ai_likelihood": round(ai_likelihood, 4),
    }


async def verify_text_claim(query_text: str) -> dict:
    """
    Verify a text claim using Google FactCheck API (when available)
    plus offline stylometric analysis for AI-generated text detection.
    """
    # --- Always compute stylometrics (works offline) ---
    stylometrics = _compute_stylometrics(query_text)

    # --- Try FactCheck API ---
    fact_check_available = False
    matches = []
    api_error = None

    if not FACT_CHECK_API_KEY:
        api_error = "API key missing in backend/.env — using offline stylometrics only."
    else:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query_text,
            "key": FACT_CHECK_API_KEY,
            "languageCode": "en",
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params, timeout=7.0)
                if resp.status_code != 200:
                    api_error = f"Google API error HTTP {resp.status_code}"
                else:
                    data = resp.json()
                    claims = data.get("claims", [])
                    fact_check_available = True
                    for c in claims[:3]:
                        review = c.get("claimReview", [{}])[0]
                        matches.append({
                            "text": c.get("text"),
                            "claimant": c.get("claimant", "Unknown"),
                            "rating": review.get("textualRating", "Unrated"),
                            "publisher": review.get("publisher", {}).get("name", "Unknown Source"),
                            "url": review.get("url", "#"),
                        })
        except Exception as e:
            api_error = str(e)
            logger.warning(f"FactCheck API request failed: {e}")

    # --- Combined verdict ---
    has_factcheck_matches = len(matches) > 0
    stylometric_ai = stylometrics["stylometric_ai_likelihood"] >= 0.5

    # Determine overall verdict
    if has_factcheck_matches:
        verdict = "DEBUNKED"
        confidence = 0.95
        is_manipulated = True
        explanation = (
            f"Matched {len(matches)} verified fact-checking record(s). "
            f"Stylometric AI-likelihood: {round(stylometrics['stylometric_ai_likelihood'] * 100, 1)}%."
        )
    elif stylometric_ai:
        verdict = "SUSPECT_AI"
        confidence = stylometrics["stylometric_ai_likelihood"]
        is_manipulated = True
        explanation = (
            f"No fact-check records matched, but text shows AI-generated stylometric patterns "
            f"(burstiness: {stylometrics['burstiness']}, TTR: {stylometrics['ttr']}, "
            f"transition density: {stylometrics['transition_density']} per 100 words). "
            f"Likely LLM-generated or AI-assisted misinformation."
        )
    else:
        verdict = "UNVERIFIED"
        confidence = 0.5
        is_manipulated = False
        explanation = "No verified debunking records found and no strong AI-stylometric signals detected."

    return {
        "verdict": verdict,
        "confidence": confidence,
        "is_manipulated": is_manipulated,
        "plain_explanation": explanation,
        "fact_check_available": fact_check_available,
        "fact_check_error": api_error,
        "matches": matches,
        "stylometrics": stylometrics,
    }


# Backwards-compatible alias
analyze_text_forensics = verify_text_claim
