# AutoAce Voice Tone & Background Noise Trial — Technical Memo

**Hosted dashboard**: https://sanchitk8-autoace-voice-tone.hf.space
**Login**: `autoace_reviewer` / `bwrTZfhLcYd3qpTr`
**Code**: https://github.com/sanchitkakarla/Aero 

## 1. Objective recap

Classify emotional tone and background-noise/audio-quality characteristics for production call audio, matching the schema in the trial spec, staying under $0.003/audio-minute, and shipping a hosted dashboard for batch evaluation.

## 2. Approaches tested

Two materially different signal sources feed the pipeline, plus one approach that was tried and discarded.

### 2a. Acoustic features (deterministic, no ML) — `features.py`

Everything that's physically measurable directly from the waveform is handled without any model:

- **Noise presence/severity**: segment-wise noise floor (dBFS), not a single global average. An earlier version used one number for the whole clip and completely missed intermittent background noise (a global SNR estimate gets washed out once speech is louder than the noise bed). Switching to "noisiest 3-second window wins" fixed this.
- **Audio quality**: clipping ratio + SNR + spectral flatness.
- **Long silence**: `librosa.effects.split` gap detection, threshold tuned to 8s after an early version (2–4.5s) flagged normal conversational pauses as anomalies.
- **Speaker overlap**: originally planned as stereo channel-correlation (if L/R disagree, it's two separate mic feeds and simultaneous activity = overlap). This turned out to be a dead end — see 3b below.

### 2b. Emotional tone — two competing models

**Approach A — acoustic SER** (`emotion_acoustic.py`): `superb/wav2vec2-base-superb-er`, a wav2vec2 model fine-tuned on IEMOCAP (4 classes: neutral/happy/angry/sad), run directly on the waveform. Mapped onto the 5-class AutoAce schema with confidence-based intensity.

**Approach B — transcript sentiment** (`emotion_text.py`): local Whisper (`base`) transcribes the call, then `j-hartmann/emotion-english-distilroberta-base` classifies the transcript text (Ekman-style 7-class output, mapped onto the 5-class schema).

**Result on the 3 labeled calls**: Approach A got 0/3 correct — it defaulted to "angry" on every call, including the ones labeled `neutral` and `satisfied`. It's picking up call-center vocal energy/pace as anger, which is exactly the failure mode the spec warns against ("do not infer frustration or distress solely from loudness"). Approach B got 2/3 correct and was adopted as the primary signal.

Approach A originally ran alongside Approach B in production, folded in only as a confidence booster (if both models agreed, confidence went up; it never overrode the text model's label). It's now **off by default** (`LOAD_ACOUSTIC_EMOTION=1` to re-enable) — since it never influenced the actual `emotional_tone`/`emotional_intensity` output and measured 0/3 standalone, keeping a whole extra model resident in memory for a minor confidence-calibration nudge wasn't worth the cost once memory became a real constraint on the hosting side. Validated that removing it doesn't change any field's accuracy on the labeled set.

**Approach C — valence/arousal ensemble** (prototyped, not included in the final repo): projects both models' class probabilities onto a shared valence/arousal space and averages, on the theory that blending would smooth out each model's individual blind spots (acoustic misses calm-but-negative calls, text misses angry-but-plain-worded calls). Tested against the same 3 calls: it *did* fix the one text-only miss (call_001, garbled transcript, correctly identified as negative-high-arousal via the acoustic signal) but broke a call the text model had gotten right (call_002) and left emotional_intensity accuracy at 0/3 (down from 3/3 for text-only). Net effect was negative on this sample. Documenting this as evidence of process, not as a recommendation — the honest read is that 3 examples are nowhere near enough to fit a 2-parameter blend without just chasing noise. Not carried into the shipped pipeline since it isn't wired into `pipeline.py` and doesn't outperform the primary approach; the idea (and what it would take to validate it properly) is captured here for future work.

### 2c. Speaker overlap — diarization, not channel correlation

Initial plan was stereo channel-correlation overlap detection. Checking the actual files: `call_001.ogg`'s left/right channels have correlation 0.9999997 — this is mono audio duplicated to stereo, not a real two-track caller/agent recording. All 3 sample files are like this. There's no channel-separation signal to exploit.

Replaced with real speaker diarization (`pyannote/speaker-diarization-3.1` via `overlap.py`), which segments the mixed-down audio into per-speaker turns and flags overlap when two turns' time ranges intersect by ≥0.8s. This requires a (free) Hugging Face token and accepting the model's usage terms — documented in the README as a setup step. Result: 3/3 correct on the labeled calls (up from a 1/3 default-false baseline). `features.py` detects whether the channels are duplicated and only invokes diarization (the more expensive path) when the cheap correlation check can't answer the question — real stereo call recordings would skip diarization entirely.

## 3. Final architecture

```
call_001.ogg
     │
     ├─► features.py        (acoustic: noise, silence, quality, needs_diarization flag)
     ├─► overlap.py          (pyannote diarization, only if channels are duplicated mono)
     ├─► emotion_text.py     (Whisper transcript → j-hartmann text-emotion classifier)
     └─► emotion_acoustic.py (wav2vec2-superb-er, off by default -- see below)
                │
                ▼
          pipeline.py (orchestrator — merges all of the above into the required JSON schema)
                │
                ▼
          app.py (Streamlit dashboard, hosted as a Docker Space on Hugging Face — login, ZIP+CSV batch upload, progress, CSV/JSON download)
```

Every module is independently runnable and testable from the CLI (`python features.py call.ogg`, etc.) — this was a deliberate choice so any one piece (e.g. swapping the SER model, or adding a real audio-event-tagging model for `background_noise_type`) can be replaced without touching the rest of the pipeline.

**No paid API is used anywhere.** All models run locally: Whisper (`base`, 139MB), wav2vec2-superb-er (~95MB), j-hartmann emotion-distilroberta (~330MB), pyannote diarization (~150MB across its sub-models). This was a deliberate choice for this trial (see §5, cost) and also sidesteps the "customer audio leaving AutoAce-controlled infrastructure" disclosure requirement entirely — nothing is sent anywhere.

## 4. Validation results

Ran via `validate.py` against the 3 labeled calls (leave-nothing-out, since n=3 doesn't support a held-out split — treat as a smoke test, not a statistically meaningful accuracy claim):

| Field | Accuracy |
|---|---|
| `emotional_tone` | 33% (1/3) |
| `emotional_intensity` | 100% (3/3) |
| `background_noise_present` | 100% (3/3) |
| `background_noise_severity` | 100% (3/3) |
| `audio_quality` | 100% (3/3) |
| `speaker_overlap_present` | 100% (3/3) |
| `long_silence_present` | 100% (3/3) |

Confusion matrix, `emotional_tone` (rows = actual, cols = predicted):

```
upset:     {distressed: 1}
neutral:   {neutral: 1}
satisfied: {neutral: 1}
```

`emotional_tone` is the weak point — it's a genuinely hard problem (5-class emotion from a mix of acoustic and semantic signal) and 3 examples is not enough to calibrate a 5-class boundary reliably. The two errors are both "in the right neighborhood" (distressed vs. upset are adjacent negative-high-arousal states; satisfied vs. neutral are adjacent positive/flat-valence states) rather than wild misses (e.g. never confused satisfied with distressed). All other fields, which are more directly tied to measurable acoustic properties, validated cleanly once the calibration issues in §2 were fixed.

**Caveat on all of the above**: three data points cannot support real accuracy claims, and further threshold-tuning against them risks overfitting to this exact sample rather than improving general performance (the ensemble experiment in §2b is a concrete example of that happening). Real validation has to happen against AutoAce's hidden test set.

## 5. Cost analysis

All inference is local (CPU, no GPU used or required) — no per-call API billing. To express this against the $0.003/audio-minute ceiling, the relevant cost is compute time on rented infrastructure:

- Measured processing time / audio duration ratio across the 3 calls: 0.63x, 0.41x, 0.56x real-time (average ~0.53x) — i.e., processing 1 minute of audio takes about 32 seconds of single-process CPU time on an Apple Silicon Mac (M-series, unspecified core count, no GPU acceleration, no batching).
- Assuming a modest cloud CPU instance (e.g., ~4 vCPU, ~$0.10–0.15/hr — roughly an AWS `c6i.xlarge`-class box), 32 seconds of compute costs **~$0.0009–0.0013 per audio-minute** — comfortably under the $0.003 ceiling, with room to spare for concurrent batch throughput or a safety margin on slower instance types.
- This estimate is single-threaded and unoptimized. Production would batch multiple files concurrently (the models are already loaded once and reused across a whole batch in `pipeline.py`/`app.py`, so per-file marginal cost drops further once amortized past the first file) and could quantize models (int8) for another 2-4x speedup if needed.
- No external paid API is used, so there's no per-request billing, rate limits, or data-retention disclosure required under the trial's external-API clause.

**Hosting cost, separate from the per-minute inference estimate above**: the dashboard is deployed on Hugging Face Spaces (Docker, `cpu-basic` hardware) under a Pro subscription, $9/month flat. That's not part of the $0.003/audio-minute ceiling — it's the cost of having the app *sitting there and reachable* at all, independent of how much audio actually gets processed through it. Went with a paid tier after Streamlit Cloud's free tier (the original plan) couldn't reliably hold this pipeline's four models in memory at once — see `APPROACH.md` for the full account of that debugging process. Worth noting this as a real operational cost AutoAce would want in the loop even though it's outside the per-minute compute estimate the ceiling is scoped to.

## 6. Latency analysis

Per-clip wall-clock time (includes Whisper transcription + text classification + acoustic feature extraction + diarization when triggered), measured on this dev machine:

| Call | Duration | Processing time | Ratio |
|---|---|---|---|
| call_001.ogg | 30.9s | 19.5s | 0.63x realtime |
| call_002.ogg | 35.0s | 14.4s | 0.41x realtime |
| call_003.ogg | 171.9s | 95.9s | 0.56x realtime |

All three process faster than real-time, which is what matters for production batch analysis (the spec doesn't require live/streaming latency). Model loading (Whisper + both classifiers + diarization pipeline) happens once per process and is cached for the rest of the batch — the numbers above are steady-state per-file cost, not including the one-time ~15-25s cold start the first time the dashboard processes a batch after starting up.

## 7. Failure modes, limitations, next steps

**Known limitations, honestly:**
- `emotional_tone` accuracy (1/3 on the labeled set) is the biggest gap. With only 3 labels there's no responsible way to further calibrate without overfitting — this needs either more labeled data or a better base model (a larger multimodal model fine-tuned specifically for call-center emotion would likely help significantly).
- `background_noise_type` is a weak heuristic (spectral flatness → "static/hiss" vs "chatter/media"), not a real classifier. An audio-event-tagging model (PANNs/YAMNet-style) would give an actual noise-type label instead of a coarse two-bucket guess. Flagged in the code as a documented TODO, not shipped due to time.
- Speaker-overlap detection depends on `pyannote.audio`, which requires a Hugging Face account + accepting model terms — an operational dependency (not a cost one) that needs to be documented for whoever deploys this in production.
- All threshold tuning (noise severity cutoffs, silence duration, overlap ratio) was calibrated against 3 examples. That's enough to catch outright bugs (which it did — multiple times) but not enough to trust the exact threshold values in production without validating against a larger set.
- Whisper occasionally hallucinates on quiet, silence-heavy, or non-English audio (seen directly on call_002, which is partly in Spanish and produced a garbled transcript fragment) — this degrades the text-based emotion signal specifically for calls where audio quality is already poor, which is an unfortunate correlation (exactly the calls hardest to classify are also the ones most likely to break transcription).

**Next steps with more time/data:**
1. Get labeled data beyond 3 examples — even 30-50 calls would allow real train/validation splitting and meaningfully better threshold calibration.
2. Swap in a proper audio-event-tagging model for `background_noise_type`.
3. Revisit the valence/arousal ensemble (§2b) once there's enough data to validate it isn't just overfitting.
4. Consider a larger/better emotion model specifically fine-tuned on call-center or customer-service audio rather than acted dialogue (IEMOCAP) or general text (Ekman-6 news/social data) — both underlying models are trained on out-of-domain data relative to actual customer calls.
