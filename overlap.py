"""Speaker overlap via real diarization (pyannote), used when the audio
doesn't have separate per-speaker channels to lean on (see features.py's
channels_are_duplicated -- true for all 3 sample calls, so this is the
only path that actually works for this dataset).
"""
import os
import tempfile

import soundfile as sf
from dotenv import load_dotenv

import features

load_dotenv()

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        token = os.getenv("HF_TOKEN")
        _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    return _pipeline


def detect(path, min_overlap_sec=0.8):
    pipeline = _get_pipeline()

    # pyannote's chunked reader chokes seeking around variable-bitrate
    # opus/ogg (sample counts drift off by a handful per chunk); a flat
    # wav sidesteps that entirely
    mono, _ = features.load_audio(path)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, mono, features.SR)
        diarization = pipeline(tmp.name)

    annotation = diarization.speaker_diarization
    turns = [(turn.start, turn.end) for turn, _, _ in annotation.itertracks(yield_label=True)]
    turns.sort()

    overlap_total = 0.0
    for i in range(len(turns)):
        for j in range(i + 1, len(turns)):
            start = max(turns[i][0], turns[j][0])
            end = min(turns[i][1], turns[j][1])
            if end > start:
                overlap_total += end - start
            if turns[j][0] > turns[i][1]:
                break  # sorted by start, no more overlaps possible for turn i

    return overlap_total >= min_overlap_sec, overlap_total


if __name__ == "__main__":
    import sys
    present, seconds = detect(sys.argv[1])
    print(f"overlap_present={present} total_overlap_sec={seconds:.2f}")
