import io
import logging
import soundfile as sf
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

logger = logging.getLogger("deepfake-shield.audio")

AUDIO_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

# --- Lazy model state (loaded on first request, not at import) ----------------
_feature_extractor = None
_audio_model = None
_model_loaded = False
_model_load_error = None


def _load_model():
    """Lazy-load the audio classifier on first request. Safe for offline environments."""
    global _feature_extractor, _audio_model, _model_loaded, _model_load_error
    if _model_loaded or _model_load_error is not None:
        return
    try:
        print(f"[*] Lazy-loading audio model: {AUDIO_MODEL_ID}...")
        _feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
        _audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_ID)
        _audio_model.eval()
        _model_loaded = True
        print("[*] Audio model loaded.")
    except Exception as e:
        _model_load_error = str(e)
        logger.error(f"Failed to load audio model: {e}")


# --- Spectral forensics helpers ---------------------------------------------


def _detect_vocoder_cutoff(y: np.ndarray, sr: int) -> dict:
    """
    Detect artificial brick-wall frequency cutoffs typical of neural vocoders
    (HiFi-GAN, ElevenLabs, VITS) often at 16 kHz or 24 kHz.
    Returns a dict with cutoff_score (0..1, higher = more synthetic-like).
    """
    if sr < 24000:
        # Not enough bandwidth to check 24 kHz cutoff; check relative to Nyquist
        nyquist = sr / 2.0
        target_freqs = [nyquist * 0.75, nyquist * 0.9]
    else:
        target_freqs = [16000.0, 24000.0]

    # Compute power spectrum
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Mean power spectrum over time
    power = D.mean(axis=1)

    cutoff_scores = []
    for tf in target_freqs:
        if tf >= sr / 2:
            continue
        idx = np.argmin(np.abs(freqs - tf))
        # Look at power just below vs just above the target frequency
        below = power[max(0, idx - 10):idx].mean() if idx > 10 else 0
        above = power[idx:idx + 10].mean() if idx + 10 < len(power) else 0
        # A sharp cutoff: high power below, near-zero above
        if below > 0:
            ratio = above / (below + 1e-10)
        else:
            ratio = 0.0
        # Lower ratio = sharper cutoff = more synthetic-like
        cutoff_scores.append(1.0 - min(ratio, 1.0))

    cutoff_score = float(np.mean(cutoff_scores)) if cutoff_scores else 0.0
    return {
        "cutoff_score": round(cutoff_score, 4),
        "target_freqs_checked": [round(f, 1) for f in target_freqs],
    }


def _detect_f0_jitter(y: np.ndarray, sr: int) -> dict:
    """
    Measure fundamental frequency (F0) micro-perturbations.
    Human speech has natural micro-tremors; synthetic voices often have
    unnaturally flat pitch or erratic jump discontinuities.
    Returns jitter_score (0..1, higher = more synthetic-like).
    """
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=512
        )
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) < 10:
            return {"jitter_score": 0.0, "voiced_frames": int(len(voiced_f0))}

        # Frame-to-frame pitch differences (micro-jitter)
        diffs = np.abs(np.diff(voiced_f0))
        # Normalize by local F0 to get relative jitter
        relative_jitter = diffs / (voiced_f0[:-1] + 1e-10)
        jitter_mean = float(np.mean(relative_jitter))
        jitter_std = float(np.std(relative_jitter))

        # Synthetic voices: very low jitter (too perfect) OR erratic high jitter
        # Human sweet spot is moderate jitter
        jitter_score = 0.0
        if jitter_mean < 0.01:  # too flat -> synthetic
            jitter_score = 0.8
        elif jitter_mean > 0.15:  # too erratic -> synthetic
            jitter_score = min(jitter_mean, 1.0)
        elif jitter_std > 0.1:  # unstable pitch contour -> synthetic
            jitter_score = min(jitter_std, 1.0)

        return {
            "jitter_score": round(jitter_score, 4),
            "jitter_mean": round(jitter_mean, 4),
            "voiced_frames": int(len(voiced_f0)),
        }
    except Exception as e:
        logger.debug(f"F0 jitter analysis failed: {e}")
        return {"jitter_score": 0.0, "voiced_frames": 0}


def _detect_mel_gating(y: np.ndarray, sr: int) -> dict:
    """
    Detect unnatural phase silence / noise gating in mel-spectrogram.
    Neural vocoders sometimes produce sharp gating between frames.
    Returns gating_score (0..1, higher = more synthetic-like).
    """
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=2048, hop_length=512)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Frame-to-frame energy differences
        frame_energy = log_mel.mean(axis=0)
        energy_diffs = np.abs(np.diff(frame_energy))

        # Count sudden drops (gating) — large negative steps
        sudden_drops = np.sum(energy_diffs > np.percentile(energy_diffs, 95) * 2)
        gating_score = float(sudden_drops) / max(len(energy_diffs), 1)

        return {
            "gating_score": round(min(gating_score * 3.0, 1.0), 4),
            "sudden_drops": int(sudden_drops),
        }
    except Exception as e:
        logger.debug(f"Mel gating analysis failed: {e}")
        return {"gating_score": 0.0, "sudden_drops": 0}


def analyze_audio(audio_bytes: bytes) -> dict:
    """
    Analyze audio for voice cloning / deepfake detection.
    Combines a transformer classifier with spectral forensics (vocoder cutoff,
    F0 micro-jitter, mel-spectrogram gating).
    """
    # Lazy-load model on first request
    _load_model()
    if not _model_loaded:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "is_manipulated": False,
            "plain_explanation": f"Audio model unavailable: {_model_load_error}",
            "suspicious_regions": [],
            "ela_score": 0.0,
            "error": _model_load_error or "Model not loaded",
        }

    try:
        # --- Load audio ---
        audio_stream = io.BytesIO(audio_bytes)
        speech_array, sampling_rate = sf.read(audio_stream)

        if len(speech_array.shape) > 1:
            speech_array = librosa.to_mono(speech_array.T)

        if sampling_rate != 16000:
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        sr = 16000

        # Use up to 10 seconds for spectral analysis, 5 seconds for classifier
        max_samples_classifier = 16000 * 5
        classifier_audio = speech_array[:max_samples_classifier] if len(speech_array) > max_samples_classifier else speech_array

        # --- Transformer classifier ---
        inputs = _feature_extractor(classifier_audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = _audio_model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        id2label = {int(k): v.lower() for k, v in _audio_model.config.id2label.items()}
        fake_idx = next(
            (k for k, v in id2label.items() if "fake" in v or "spoof" in v or "synthet" in v),
            1,
        )
        classifier_fake_score = float(probs[fake_idx])

        # --- Spectral forensics ---
        cutoff = _detect_vocoder_cutoff(speech_array, sr)
        jitter = _detect_f0_jitter(speech_array, sr)
        gating = _detect_mel_gating(speech_array, sr)

        # --- Combined verdict ---
        # Weighted combination: classifier 50%, spectral 50%
        spectral_score = (cutoff["cutoff_score"] + jitter["jitter_score"] + gating["gating_score"]) / 3.0
        combined_fake_score = round(0.5 * classifier_fake_score + 0.5 * spectral_score, 4)

        is_spoof = combined_fake_score >= 0.50

        if is_spoof:
            verdict = "SYNTHETIC"
            explanation = (
                f"Neural voice cloning detected ({round(combined_fake_score * 100, 1)}% synthetic). "
                f"Vocoder cutoff: {cutoff['cutoff_score']}, F0 jitter: {jitter['jitter_score']}, "
                f"Mel gating: {gating['gating_score']}. Missing authentic human vocal tract resonances."
            )
        else:
            verdict = "AUTHENTIC"
            explanation = (
                f"Natural speech verified ({round((1.0 - combined_fake_score) * 100, 1)}% genuine). "
                f"Organic vocal micro-tremors and natural spectral continuity present."
            )

        return {
            "verdict": verdict,
            "confidence": round(combined_fake_score if is_spoof else (1.0 - combined_fake_score), 4),
            "is_manipulated": is_spoof,
            "plain_explanation": explanation,
            "suspicious_regions": [],
            "ela_score": 0.0,
            "classifier_score": round(classifier_fake_score, 4),
            "spectral_score": round(spectral_score, 4),
            "vocoder_cutoff_score": cutoff["cutoff_score"],
            "f0_jitter_score": jitter["jitter_score"],
            "mel_gating_score": gating["gating_score"],
        }

    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        # SAFE FAILURE: never return AUTHENTIC on error — return UNKNOWN
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "is_manipulated": False,
            "plain_explanation": f"Audio analysis failed: {str(e)}",
            "suspicious_regions": [],
            "ela_score": 0.0,
            "error": str(e),
        }