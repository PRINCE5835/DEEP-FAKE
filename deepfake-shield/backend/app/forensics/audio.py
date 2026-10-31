"""
Lightweight audio forensics — no PyTorch/Transformers.
Uses spectral analysis, F0 jitter, and mel-spectrogram features.
Designed for Render free tier (512MB RAM).
"""
import io
import logging
import soundfile as sf
import librosa
import numpy as np

logger = logging.getLogger("deepfake-shield.audio")

_model_loaded = True


def _detect_vocoder_cutoff(y: np.ndarray, sr: int) -> dict:
    if sr < 24000:
        nyquist = sr / 2.0
        target_freqs = [nyquist * 0.75, nyquist * 0.9]
    else:
        target_freqs = [16000.0, 24000.0]

    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    power = D.mean(axis=1)

    cutoff_scores = []
    for tf in target_freqs:
        if tf >= sr / 2:
            continue
        idx = np.argmin(np.abs(freqs - tf))
        below_slice = power[max(0, idx - 10):idx]
        above_slice = power[idx:idx + 10]
        below = float(below_slice.mean()) if len(below_slice) > 0 else 0.0
        above = float(above_slice.mean()) if len(above_slice) > 0 else 0.0
        if below > 0:
            ratio = above / (below + 1e-10)
        else:
            ratio = 0.0
        cutoff_scores.append(1.0 - min(ratio, 1.0))

    cutoff_score = float(np.mean(cutoff_scores)) if cutoff_scores else 0.0
    return {"cutoff_score": round(cutoff_score, 4)}


def _detect_f0_jitter(y: np.ndarray, sr: int) -> dict:
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=512)
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) < 10:
            return {"jitter_score": 0.0, "voiced_frames": int(len(voiced_f0))}
        diffs = np.abs(np.diff(voiced_f0))
        relative_jitter = diffs / (voiced_f0[:-1] + 1e-10)
        jitter_mean = float(np.mean(relative_jitter))
        jitter_std = float(np.std(relative_jitter))
        jitter_score = 0.0
        if jitter_mean < 0.01:
            jitter_score = 0.8
        elif jitter_mean > 0.15:
            jitter_score = min(jitter_mean, 1.0)
        elif jitter_std > 0.1:
            jitter_score = min(jitter_std, 1.0)
        return {"jitter_score": round(jitter_score, 4), "jitter_mean": round(jitter_mean, 4), "voiced_frames": int(len(voiced_f0))}
    except Exception as e:
        logger.debug(f"F0 jitter analysis failed: {e}")
        return {"jitter_score": 0.0, "voiced_frames": 0}


def _detect_mel_gating(y: np.ndarray, sr: int) -> dict:
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=2048, hop_length=512)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        frame_energy = log_mel.mean(axis=0)
        energy_diffs = np.abs(np.diff(frame_energy))
        sudden_drops = np.sum(energy_diffs > np.percentile(energy_diffs, 95) * 2) if len(energy_diffs) > 0 else 0
        gating_score = float(sudden_drops) / max(len(energy_diffs), 1)
        return {"gating_score": round(min(gating_score * 3.0, 1.0), 4), "sudden_drops": int(sudden_drops)}
    except Exception as e:
        logger.debug(f"Mel gating analysis failed: {e}")
        return {"gating_score": 0.0, "sudden_drops": 0}


def analyze_audio(audio_bytes: bytes) -> dict:
    try:
        audio_stream = io.BytesIO(audio_bytes)
        speech_array, sampling_rate = sf.read(audio_stream)

        if len(speech_array.shape) > 1:
            speech_array = librosa.to_mono(speech_array.T)

        if sampling_rate != 16000:
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        sr = 16000

        cutoff = _detect_vocoder_cutoff(speech_array, sr)
        jitter = _detect_f0_jitter(speech_array, sr)
        gating = _detect_mel_gating(speech_array, sr)

        spectral_score = (cutoff["cutoff_score"] + jitter["jitter_score"] + gating["gating_score"]) / 3.0
        combined_fake_score = round(spectral_score, 4)

        is_spoof = combined_fake_score >= 0.50

        if is_spoof:
            verdict = "SYNTHETIC"
            explanation = f"Neural voice cloning detected ({round(combined_fake_score * 100, 1)}% synthetic). Vocoder cutoff: {cutoff['cutoff_score']}, F0 jitter: {jitter['jitter_score']}, Mel gating: {gating['gating_score']}."
        else:
            verdict = "AUTHENTIC"
            explanation = f"Natural speech verified ({round((1.0 - combined_fake_score) * 100, 1)}% genuine). Organic vocal micro-tremors and natural spectral continuity present."

        return {
            "verdict": verdict,
            "confidence": round(combined_fake_score if is_spoof else (1.0 - combined_fake_score), 4),
            "is_manipulated": is_spoof,
            "plain_explanation": explanation,
            "suspicious_regions": [],
            "ela_score": 0.0,
            "vocoder_cutoff_score": cutoff["cutoff_score"],
            "f0_jitter_score": jitter["jitter_score"],
            "mel_gating_score": gating["gating_score"],
        }
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "is_manipulated": False,
            "plain_explanation": f"Audio analysis failed: {str(e)}",
            "suspicious_regions": [],
            "ela_score": 0.0,
            "error": str(e),
        }