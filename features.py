"""Deterministic acoustic feature extraction from call audio.

Covers everything that's physically measurable from the waveform, so we
don't need to burn LLM calls on it: noise, silence, overlap, clipping/quality.
"""
import numpy as np
import librosa

SR = 16000  # downsample everything to 16k, plenty for these features and cuts load time


def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=False)
    if y.ndim == 2:
        # keep stereo around for overlap detection, but also build a mono mix
        stereo = y
        mono = librosa.to_mono(y)
    else:
        stereo = None
        mono = y
    return mono, stereo


def frame_rms(y, frame_length=2048, hop_length=512):
    return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]


def estimate_snr_db(y):
    """Rough SNR estimate: treat the quietest 10% of frames as noise floor,
    loudest 90% as signal. Not rigorous but good enough to rank clips."""
    rms = frame_rms(y)
    rms = rms[rms > 0]
    if len(rms) < 10:
        return 40.0
    sorted_rms = np.sort(rms)
    noise_floor = np.mean(sorted_rms[: max(1, len(sorted_rms) // 10)])
    signal_level = np.mean(sorted_rms[int(len(sorted_rms) * 0.5):])
    if noise_floor <= 1e-6:
        return 40.0
    return 20 * np.log10(signal_level / noise_floor)


def noise_floor_dbfs(y):
    """Absolute loudness of the quietest 10% of frames, relative to full scale.
    Unlike SNR this doesn't get washed out when speech is loud -- a noisy
    background stays visible even under a strong voice."""
    rms = frame_rms(y)
    rms = rms[rms > 0]
    if len(rms) < 10:
        return -80.0
    sorted_rms = np.sort(rms)
    noise_floor = np.mean(sorted_rms[: max(1, len(sorted_rms) // 10)])
    if noise_floor <= 1e-8:
        return -80.0
    return 20 * np.log10(noise_floor)


def detect_long_silence(y, sr=SR, silence_thresh_db=-40, min_silence_sec=8.0):
    intervals = librosa.effects.split(y, top_db=-silence_thresh_db)
    if len(intervals) == 0:
        return True, len(y) / sr

    longest_gap = 0.0
    prev_end = 0
    for start, end in intervals:
        gap = (start - prev_end) / sr
        longest_gap = max(longest_gap, gap)
        prev_end = end
    tail_gap = (len(y) - prev_end) / sr
    longest_gap = max(longest_gap, tail_gap)

    return longest_gap >= min_silence_sec, longest_gap


def spectral_flatness_mean(y):
    return float(np.mean(librosa.feature.spectral_flatness(y=y)))


def estimate_clipping_ratio(y):
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak <= 0:
        return 0.0
    clipped = np.sum(np.abs(y) >= 0.98 * peak)
    return clipped / len(y)


def channels_are_duplicated(stereo, corr_thresh=0.999):
    """Some of these files are mono audio duplicated across both stereo
    channels rather than a real per-speaker stereo recording. When that's
    true there's no channel-separation signal to exploit for overlap
    detection -- has to come from diarization instead."""
    if stereo is None or stereo.shape[0] < 2:
        return True
    left, right = stereo[0], stereo[1]
    n = min(len(left), len(right))
    if n == 0:
        return True
    corr = np.corrcoef(left[:n], right[:n])[0, 1]
    return corr > corr_thresh


def detect_overlap_from_stereo(stereo, sr=SR):
    """Channel-correlation based overlap detection -- only meaningful when
    left/right are genuinely separate speaker tracks. Returns None when the
    channels are duplicated (caller code should fall back to diarization)."""
    if stereo is None or stereo.shape[0] < 2:
        return None
    if channels_are_duplicated(stereo):
        return None

    left, right = stereo[0], stereo[1]
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]

    l_peak = np.max(np.abs(left)) or 1e-6
    r_peak = np.max(np.abs(right)) or 1e-6
    active_thresh = 0.15

    win = sr
    overlap_frames = 0
    total_frames = 0
    for i in range(0, n - win, win):
        l, r = left[i:i + win], right[i:i + win]
        l_active = np.sqrt(np.mean(l ** 2)) > active_thresh * l_peak
        r_active = np.sqrt(np.mean(r ** 2)) > active_thresh * r_peak
        if l_active or r_active:
            total_frames += 1
            if l_active and r_active:
                overlap_frames += 1

    if total_frames == 0:
        return False
    return (overlap_frames / total_frames) > 0.4


def segment_noise_floors(y, sr=SR, win_sec=3.0):
    """Noise floor computed per window instead of over the whole clip, so
    noise that only shows up in part of the call (TV in the background for
    30s, say) doesn't get averaged out by the quiet parts."""
    win = int(win_sec * sr)
    floors = []
    for i in range(0, max(1, len(y) - win), win):
        seg = y[i:i + win]
        if len(seg) < win // 2:
            continue
        floors.append(noise_floor_dbfs(seg))
    return floors or [noise_floor_dbfs(y)]


def classify_noise_severity(segment_floors, flatness):
    # take the noisiest window as representative -- that's the part a human
    # listener would actually notice and call "background noise"
    worst = max(segment_floors)
    if worst < -55:
        return "none"
    if worst < -50:
        return "low"
    if worst < -20 or flatness > 0.15:
        return "medium"
    return "high"


def classify_audio_quality(snr_db, clipping_ratio, flatness):
    if clipping_ratio > 0.02 or snr_db < 8:
        return "severely_impaired"
    if clipping_ratio > 0.002 or snr_db < 18:
        return "slightly_impaired"
    return "clear"


def extract(path):
    mono, stereo = load_audio(path)
    duration = len(mono) / SR

    snr_db = estimate_snr_db(mono)
    seg_floors = segment_noise_floors(mono)
    floor_db = max(seg_floors)
    clipping = estimate_clipping_ratio(mono)
    flatness = spectral_flatness_mean(mono)
    long_silence, longest_gap = detect_long_silence(mono)
    overlap = detect_overlap_from_stereo(stereo)
    needs_diarization = overlap is None  # duplicated/mono channels -> no usable stereo signal

    severity = classify_noise_severity(seg_floors, flatness)
    noise_present = severity != "none"
    quality = classify_audio_quality(snr_db, clipping, flatness)

    return {
        "duration_sec": duration,
        "snr_db": snr_db,
        "noise_floor_dbfs_worst_segment": floor_db,
        "clipping_ratio": clipping,
        "spectral_flatness": flatness,
        "background_noise_present": noise_present,
        "background_noise_severity": severity,
        "audio_quality": quality,
        "speaker_overlap_present": bool(overlap) if overlap is not None else False,
        "needs_diarization_for_overlap": needs_diarization,
        "long_silence_present": long_silence,
        "longest_silence_gap_sec": longest_gap,
    }


if __name__ == "__main__":
    import sys
    import json
    print(json.dumps(extract(sys.argv[1]), indent=2, default=float))
