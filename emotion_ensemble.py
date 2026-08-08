"""Combines the acoustic and text-based emotion models by projecting both
onto a shared valence (negative<->positive) / arousal (calm<->activated)
space and averaging, then mapping the blended point back onto AutoAce's
5-class schema.

Neither model alone is reliable here: acoustic SER conflates loud/fast
call-center speech with anger (misses satisfied/neutral entirely), and the
text model misses paralinguistic tone when the transcript is short, garbled,
or just doesn't contain emotionally loaded words even though the delivery
does. Averaging in VA space is a coarse fix but pulls both failure modes
toward the middle instead of committing fully to either one's mistake.
"""
import emotion_acoustic
import emotion_text

# rough valence/arousal anchors per source label, on a [-1, 1] scale
ACOUSTIC_VA = {
    "neu": (0.0, -0.2),
    "hap": (0.7, 0.5),
    "ang": (-0.7, 0.8),
    "sad": (-0.6, -0.4),
}

TEXT_VA = {
    "neutral": (0.0, -0.2),
    "joy": (0.7, 0.4),
    "anger": (-0.7, 0.7),
    "sadness": (-0.5, -0.3),
    "disgust": (-0.6, 0.2),
    "fear": (-0.4, 0.6),
    "surprise": (0.2, 0.6),
}


def _va_from_scores(raw_scores, va_table):
    v, a, total = 0.0, 0.0, 0.0
    for label, score in raw_scores.items():
        if label not in va_table:
            continue
        lv, la = va_table[label]
        v += lv * score
        a += la * score
        total += score
    if total == 0:
        return 0.0, -0.2
    return v / total, a / total


def _va_to_label(valence, arousal):
    if abs(valence) < 0.15 and arousal < 0.1:
        tone = "neutral"
    elif valence > 0:
        tone = "satisfied"
    elif arousal > 0.35:
        tone = "distressed" if arousal > 0.6 else "upset"
    else:
        tone = "frustrated"

    intensity = "high" if arousal > 0.55 or abs(valence) > 0.6 else ("medium" if arousal > 0.15 or abs(valence) > 0.3 else "low")
    return tone, intensity


def classify(path):
    acoustic = emotion_acoustic.classify(path)
    text = emotion_text.classify(path, transcript=None)

    v_a, a_a = _va_from_scores(acoustic["_raw_scores"], ACOUSTIC_VA)
    v_t, a_t = _va_from_scores(text["_raw_scores"], TEXT_VA)

    # weight text higher -- it beat acoustic-only 2/3 vs 0/3 on the labeled set
    valence = 0.35 * v_a + 0.65 * v_t
    arousal = 0.35 * a_a + 0.65 * a_t

    tone, intensity = _va_to_label(valence, arousal)
    agreement = acoustic["emotional_tone"] == text["emotional_tone"]
    confidence = max(acoustic["confidence"], text["confidence"])
    if agreement:
        confidence = min(1.0, confidence + 0.1)

    return {
        "emotional_tone": tone,
        "emotional_intensity": intensity,
        "confidence": round(confidence, 2),
        "_valence": round(valence, 3),
        "_arousal": round(arousal, 3),
        "_acoustic": {"tone": acoustic["emotional_tone"], "scores": acoustic["_raw_scores"]},
        "_text": {"tone": text["emotional_tone"], "transcript": text.get("_transcript", "")},
    }


if __name__ == "__main__":
    import sys
    import json
    print(json.dumps(classify(sys.argv[1]), indent=2))
