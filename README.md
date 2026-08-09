---
title: AutoAce Voice Tone Dashboard
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AutoAce Voice Tone & Background Noise Dashboard

Classifies emotional tone and background-noise/audio-quality characteristics for call recordings. See `MEMO.md` for the technical writeup (approach, validation results, cost/latency analysis) and `APPROACH.md` for the build narrative.

## What's here

- `features.py` — deterministic acoustic analysis (noise, silence, audio quality)
- `overlap.py` — speaker-overlap detection via `pyannote` diarization
- `emotion_text.py` — primary emotional-tone model (Whisper transcript + text-emotion classifier)
- `emotion_acoustic.py` — secondary tone-of-voice model, off by default (see `MEMO.md` §2b)
- `pipeline.py` — orchestrator, produces the required JSON schema per clip
- `validate.py` — runs the pipeline against `data/labels.csv` and prints a scorecard + confusion matrix
- `app.py` — the Streamlit dashboard

## Running locally

Requires Python 3.12 and `ffmpeg` installed on your system (`brew install ffmpeg` on Mac).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_read_token
APP_USER=choose_a_username
APP_PASS=choose_a_password
```

`HF_TOKEN` needs read access and needs you to have accepted the usage terms for `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, and `pyannote/speaker-diarization-community-1` on huggingface.co (each is a one-click "agree" on the model page, required once per account).

Run the dashboard:

```bash
streamlit run app.py
```

Or run the pipeline directly against a single file from the command line:

```bash
python pipeline.py path/to/call.ogg
```

Or validate against the labeled sample calls:

```bash
python validate.py
```

## Using the dashboard

1. Log in with the credentials set in `APP_USER`/`APP_PASS`.
2. Upload a `.zip` containing audio files (`.ogg`/`.wav`/`.mp3`/`.m4a`/`.flac`) and a `labels.csv`-style manifest (`name`, `result_json` columns) at the root — see `data/labels.csv` for the format. `result_json` can be empty for unlabeled evaluation batches.
3. Click "Run batch analysis."
4. Review results in the table, download as CSV or JSON.

## Deployment

Configured to run as a Docker Space on Hugging Face (`Dockerfile` in this repo). Set the same three environment variables (`HF_TOKEN`, `APP_USER`, `APP_PASS`) as Space secrets rather than a `.env` file.
