"""Speech emotion recognition, local model only (no paid API).

Uses superb/wav2vec2-base-superb-er (trained on IEMOCAP: neutral/happy/angry/sad)
and remaps its output onto AutoAce's 5-class schema. The source model doesn't
have a native 5th class or a separate intensity signal, so intensity is
derived from how confident the model is in its top prediction, and "distressed"
vs "upset" / "satisfied" vs "neutral" is split by the same confidence value.
This is a coarse proxy -- documented as a known limitation in the memo.
"""
import warnings

import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

import features

warnings.filterwarnings("ignore")

MODEL_ID = "superb/wav2vec2-base-superb-er"

_model = None
_extractor = None


def _get_model():
    global _model, _extractor
    if _model is None:
        _extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        _model.eval()
    return _model, _extractor


def _map_label(label, top_score):
    if label == "neu":
        tone = "neutral"
        intensity = "low" if top_score < 0.6 else "medium"
    elif label == "hap":
        tone = "satisfied"
        intensity = "medium" if top_score < 0.75 else "high"
    elif label == "ang":
        tone = "upset" if top_score < 0.8 else "distressed"
        intensity = "medium" if top_score < 0.65 else "high"
    elif label == "sad":
        tone = "frustrated" if top_score < 0.7 else "distressed"
        intensity = "medium" if top_score < 0.65 else "high"
    else:
        tone, intensity = "neutral", "low"
    return tone, intensity


def classify(path):
    model, extractor = _get_model()
    mono, _ = features.load_audio(path)

    inputs = extractor(mono, sampling_rate=features.SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    id2label = model.config.id2label
    ranked = sorted(
        ({"label": id2label[i], "score": float(p)} for i, p in enumerate(probs)),
        key=lambda x: -x["score"],
    )
    top = ranked[0]
    tone, intensity = _map_label(top["label"], top["score"])
    return {
        "emotional_tone": tone,
        "emotional_intensity": intensity,
        "confidence": round(float(top["score"]), 2),
        "_raw_scores": {r["label"]: round(r["score"], 3) for r in ranked},
    }


if __name__ == "__main__":
    import sys
    import json
    print(json.dumps(classify(sys.argv[1]), indent=2))
