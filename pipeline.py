"""Orchestrator: combines the acoustic feature module with the text-based
emotion model to produce one JSON object per clip in AutoAce's required
schema. Emotion comes from emotion_text (better accuracy on the 3 labeled
calls).

emotion_acoustic (wav2vec2 SER) is off by default -- it measured 0/3 on
the labeled calls, only ever nudged the confidence score (never overrode
the primary tone/intensity label), and costs a full extra model's worth
of memory to keep resident. Set LOAD_ACOUSTIC_EMOTION=1 to bring it back
for comparison/experimentation; the production path doesn't need it.
"""
import os
import time

import features
import emotion_text
import overlap

LOAD_ACOUSTIC_EMOTION = os.getenv("LOAD_ACOUSTIC_EMOTION", "0") == "1"


def process(path):
    start = time.time()

    acoustic = features.extract(path)
    text_emotion = emotion_text.classify(path)

    speaker_overlap = acoustic["speaker_overlap_present"]
    if acoustic["needs_diarization_for_overlap"]:
        # stereo channels are duplicated/mono -- only real diarization can
        # tell us whether two people were actually talking over each other
        speaker_overlap, _ = overlap.detect(path)

    confidence = text_emotion["confidence"]
    if LOAD_ACOUSTIC_EMOTION:
        import emotion_acoustic
        try:
            acoustic_emotion = emotion_acoustic.classify(path)
        except Exception:
            acoustic_emotion = None
        if acoustic_emotion and acoustic_emotion["emotional_tone"] == text_emotion["emotional_tone"]:
            # two independent signals agreeing is worth boosting confidence for
            confidence = min(1.0, confidence + 0.1)

    result = {
        "emotional_tone": text_emotion["emotional_tone"],
        "emotional_intensity": text_emotion["emotional_intensity"],
        "background_noise_present": bool(acoustic["background_noise_present"]),
        "background_noise_type": _noise_type_guess(acoustic),
        "background_noise_severity": acoustic["background_noise_severity"],
        "audio_quality": acoustic["audio_quality"],
        "speaker_overlap_present": bool(speaker_overlap),
        "long_silence_present": bool(acoustic["long_silence_present"]),
        "confidence": round(confidence, 2),
    }

    elapsed = time.time() - start
    return result, elapsed


def _noise_type_guess(acoustic):
    """Acoustic features can say *that* there's noise and roughly how loud
    it is, but not what it is -- we don't have an audio-event-tagging model
    wired in yet (see memo, future work). Flatness is a weak proxy: flat
    spectrum reads more like static/hiss, tonal reads more like chatter/TV."""
    if not acoustic["background_noise_present"]:
        return ""
    return "static/hiss" if acoustic["spectral_flatness"] > 0.1 else "background chatter or media"


if __name__ == "__main__":
    import sys
    import json
    result, elapsed = process(sys.argv[1])
    result["_latency_sec"] = round(elapsed, 2)
    print(json.dumps(result, indent=2))
