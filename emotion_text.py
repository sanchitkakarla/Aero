"""Approach B for emotional tone: transcribe with Whisper, classify the
words rather than the waveform. Acoustic SER (emotion_acoustic.py) picks up
raised call-center voices as "angry" regardless of what's actually being
said -- this approach should be less fooled by loudness/pace since it
reasons over the transcript text.
"""
import warnings

import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

TEXT_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

_whisper_model = None
_text_model = None
_tokenizer = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _get_text_model():
    global _text_model, _tokenizer
    if _text_model is None:
        _tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        _text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID)
        _text_model.eval()
    return _text_model, _tokenizer


def transcribe(path):
    model = _get_whisper()
    result = model.transcribe(path, fp16=False)
    return result["text"].strip()


# j-hartmann model outputs Ekman-style emotions: anger, disgust, fear, joy,
# neutral, sadness, surprise -- map onto AutoAce's 5-class schema.
def _map_label(label, score):
    if label == "neutral":
        return "neutral", "low" if score < 0.6 else "medium"
    if label == "joy":
        return "satisfied", "medium" if score < 0.75 else "high"
    if label == "anger":
        return ("distressed" if score > 0.75 else "upset"), ("high" if score > 0.6 else "medium")
    if label in ("sadness", "disgust"):
        return "frustrated", "medium" if score < 0.65 else "high"
    if label == "fear":
        return "distressed", "high"
    if label == "surprise":
        return "neutral", "medium"
    return "neutral", "low"


def classify(path, transcript=None):
    if transcript is None:
        transcript = transcribe(path)

    if not transcript:
        return {
            "emotional_tone": "neutral",
            "emotional_intensity": "low",
            "confidence": 0.3,
            "_transcript": "",
            "_raw_scores": {},
        }

    model, tokenizer = _get_text_model()
    inputs = tokenizer(transcript, return_tensors="pt", truncation=True, max_length=512)
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
        "_transcript": transcript,
        "_raw_scores": {r["label"]: round(r["score"], 3) for r in ranked},
    }


if __name__ == "__main__":
    import sys
    import json
    print(json.dumps(classify(sys.argv[1]), indent=2))
