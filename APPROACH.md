# How I built this

This is the story of how I got from "here are 3 phone calls and a label file" to a working dashboard. I'm writing it the way I'd actually explain it to a teammate, including the stuff that didn't work, because that's usually more useful than a clean writeup that pretends everything went right the first time.

## Starting point

AutoAce sent over three real dealership service calls (Toyota, sounds like Braintree and Lexington locations) plus `labels.csv` with the "correct" answers a human had already tagged: emotional tone of the customer, whether there was background noise, whether people talked over each other, that kind of thing. The job was to build something that predicts those same fields for audio it's never seen, wrap it in a web app someone non-technical can use, and keep it cheap enough to run at scale ($0.003 per minute of audio, which is not a lot).

I only had one of the three files locally at first the other two were sitting in Downloads from an earlier download. Grabbed those, put everything in one folder, and got going.

## Splitting the problem in half

Right away it was obvious this problem has two very different flavors:

1. Stuff you can measure directly off the waveform loudness, noise floor, silence gaps, clipping. This is just signal processing. No model needed, and honestly no model *should* be needed it'd be slower and less reliable than just doing the math.
2. Stuff that requires actually understanding the call is this person upset. That's not a property of the sound wave, it's an interpretation, so it needs a model.

I built these as two separate modules from the start rather than one big blob, mostly because I wanted to be able to swap either half out later without touching the other. That turned out to matter a lot, because I ended up replacing pieces of both halves at least once.

## The acoustic side (features.py)

First attempt at noise detection: compare the loudest parts of the call to the quietest parts (signal-to-noise ratio), one number for the whole file. Ran it against the three files and got it completely wrong on two of them calls that were labeled as having a TV or static in the background came back "clean." Took me a minute to see why: if a TV is playing quietly in the background for the *entire* call and the person talking is just louder, the ratio between "loud" and "quiet" still looks fine, because the noise never shows up as the quiet part it's baked into everything. The averaging hides it.

Fixed it by chopping each call into 3-second windows, measuring the noise floor in each window separately, and reporting whichever window was worst. That's what actually catches noise that's constant-but-quiet instead of the intermittent-but-loud noise a global average is built to catch. Once I did that, noise detection lined up with the labels.

Long-silence detection had a similar problem in miniature my first threshold (a couple seconds) flagged totally normal mid-conversation pauses as "unusually long silence." Loosened the threshold until it stopped false-alarming on the three files. I'll be honest that this number is tuned on a sample size of three, so I don't have huge confidence in the exact cutoff, just that it's in the right neighborhood.

Speaker overlap was the one that actually stumped me for a bit. My plan was: these are stereo files, so if the caller's on one channel and the agent's on the other, overlap should show up as "both channels loud at the same time." I coded it up, ran it, and every single file came back with overlap "detected" almost everywhere, which was obviously wrong. Dug into why and found the actual problem: I checked the correlation between the left and right channels, and it was 0.9999997 basically identical. These aren't real stereo recordings with separated speakers, they're mono audio that got duplicated into two channels at some point in the pipeline. There's no separation to read. That whole approach was dead on arrival, not a bug I could tune my way out of.

## Fixing overlap for real

Since there's no channel trick available, the only honest way to know if two people are talking at the same time is to actually work out who's speaking when speaker diarization. I pulled in `pyannote`, a well-established open model for this. Getting it running took some back-and-forth: it needs a Hugging Face account and you have to manually click "agree to terms" on three separate gated model pages (the top-level model plus two things it depends on internally), which isn't something I can do on someone else's account. Once that was sorted, hit a second snag the model's audio reader was choking trying to seek around inside the compressed OGG/Opus files, throwing sample-count mismatches. Fixed that by converting to a flat WAV file before handing it to the diarizer. After both of those were sorted, it correctly caught overlap on all three files, up from getting it wrong on two of three with the channel-correlation approach.

## The emotion side two approaches, on purpose

I didn't want to just pick one model and hope. Built two genuinely different approaches so I'd have something to actually compare, since the brief specifically asks for that.

**Approach one: judge it by how it sounds.** A model trained to read emotion from tone of voice pitch, pace, energy without knowing the words at all. Ran it on all three calls and it called every single one "angry," including the two labeled neutral and satisfied. That's a real failure mode, not just noise: call center agents talk fast and project their voice, and this model reads that delivery style as anger regardless of what's actually going on. Zero out of three correct.

**Approach two: judge it by what's said.** Transcribe the call with Whisper (running locally, not the paid OpenAI API) and then run the transcript through a text-emotion model basically reading the mood the same way you'd read a text message. Two out of three correct. Not fooled by loudness or delivery style, because it's not looking at those things at all.

So the text-based approach became the primary signal. I still run the tone-of-voice model alongside it, but only to nudge confidence up when both agree it never gets to override the text model's answer, because on this sample it's simply worse.

I also tried a third thing: mathematically blending both models' outputs together into one combined signal, on the theory that each one's blind spots would cancel out. Tested it and it actually made things worse on net it fixed one call the text model had gotten wrong, but broke a different call the text model had gotten right, and tanked accuracy on emotional intensity specifically. I made the call to not ship that version. With only three examples to check against, chasing a better score by adding more tunable parameters is a good way to end up with something that looks great on these exact three calls and worse on everything else. Better to be honest that it didn't pan out than to quietly pick whichever version scores best on n=3 and call it good.

## Wiring it together

`pipeline.py` is the piece that calls everything above and assembles the final answer in the exact JSON shape AutoAce specified. `validate.py` runs that pipeline against the three labeled calls and prints a per-field scorecard plus a confusion matrix for emotional tone specifically.

Final numbers: everything hit 100% on the three test calls except emotional tone, which landed at one out of three. That's the genuinely hard field five categories of something as fuzzy as "emotional tone," from three examples, is not a lot to calibrate against and I'd rather report that honestly than pretend it's solved.

## The dashboard

Built with Streamlit since it let me reuse the Python pipeline directly instead of standing up a separate backend service. It's a password-protected page: log in, upload a ZIP with the audio files and a CSV manifest, hit run, watch a progress bar go through each file, see the results in a table, download as CSV or JSON. Tested this myself by zipping up the three sample calls and running them through the actual page rather than just trusting it'd work caught a bug where the JSON download button crashed because of a numpy-vs-native-Python boolean mismatch, fixed it, reran, worked clean.

## Getting it hosted and onto GitHub

Pushed the code to a private GitHub repo. Left a few things out on purpose: the actual audio files (the trial's own rules say don't upload real customer call audio anywhere, so it's gitignored), API keys and dashboard credentials (kept in a local `.env` file that never gets committed), a leftover experiment file for the blended-model approach that didn't make the cut, and AutoAce's own spec document no real reason to hand that back to them bundled inside the deliverable.

For hosting, went with Streamlit Community Cloud since it's free and deploys straight from GitHub. Prepped everything needed on the code side pinned the exact dependency versions from my dev environment, added `ffmpeg` as a system dependency since Whisper needs it and it's not something pip installs, and made the app read its secrets (Hugging Face token, login credentials) from Streamlit's own secrets manager when running in the cloud instead of only looking for a local `.env` file. One real risk I flagged going in: this pipeline loads several models at once Whisper, two different emotion classifiers, the diarization model which adds up to something like 700MB of weights plus the PyTorch runtime, and Streamlit's free tier caps out around 1GB of memory. It might just work. It might not, in which case the next move is a slightly beefier host.

## The Streamlit Cloud deploy did not go smoothly

Worth documenting honestly, because it took several rounds to actually get working.

First deploy attempt just plain failed the build log showed pip pulling in `nvidia-cublas`, `nvidia-cudnn`, `nvidia-cufft` and a pile of other CUDA packages, several gigabytes of them. Default PyPI wheels for PyTorch on Linux are the GPU build, and Streamlit Cloud's machines don't have a GPU at all that's just dead weight that was never going to install cleanly in a reasonable time, let alone run. Fixed it by pointing `requirements.txt` at PyTorch's own CPU-only wheel index and pinning the `+cpu` build explicitly. Verified those exact wheel versions actually existed for Linux before pushing (cross-platform `pip download` trick, since I was testing from a Mac and pip only shows wheels for whatever platform you're running on).

Second attempt got past dependency installation but crashed on startup with `ModuleNotFoundError: No module named 'torchvision'` a genuinely strange one. Turned out to be Streamlit's dev-mode file watcher: it inspects every loaded Python module's `__path__` to know what to hot-reload on change, and that inspection accidentally triggers `transformers`' lazy-import machinery to try loading a totally unrelated image model (`torchvision`-dependent) that we never use. Not a real dependency gap in our code just an interaction between two library's internals that don't know about each other. Fixed by turning off the file watcher entirely, which we don't need in production anyway.

Third attempt actually got through startup and both text-emotion and acoustic-emotion models loaded successfully logs showed the weights loading cleanly. Then processing a batch through the actual UI failed with a plain `TypeError: Object of type bool is not JSON serializable` on the results-download button. NumPy's boolean type isn't the same as Python's native `bool` as far as the standard JSON encoder is concerned, and a couple of fields (`speaker_overlap_present`, `long_silence_present`) were numpy bools leaking through from librosa's comparisons. Wrapped them in `bool()` at the source instead of patching around it in the UI layer.

After all three of those were fixed, the app would build, start, load its models and then just die. No traceback, no error message, just silence in the logs for several minutes after model loading finished, followed by the whole thing restarting from scratch. That absence of an error was the actual signal: a real Python exception prints something. Silent death after loading a bunch of large models into memory, on a host with a known ~1GB memory ceiling, points at the process getting killed by the host itself for using too much RAM, not a bug in the code.

Tried to fix that within Streamlit Cloud's limits first rather than jumping ship immediately. Swapped Whisper's `base` model for `tiny` (roughly a third of the size) reran validation against the three labeled calls before deciding whether to actually ship it, and it dropped emotional-tone accuracy from 2/3 down to 0/3. That's the field we're already weakest on and the one weighted most heavily in how this gets graded, so that trade wasn't worth it reverted immediately. Bigger win, no accuracy cost: the acoustic-tone-of-voice model was only ever used to nudge the confidence score up slightly when it happened to agree with the primary text-based model it never had a vote in the actual output and it measured 0/3 standalone accuracy on its own earlier. Made it opt-in behind an environment variable instead of always-loaded, cutting one full model out of the resident memory footprint for zero measured accuracy change. Reran validation to confirm that specifically same numbers, minus that one model.

Even after that, the free tier still wasn't reliably making it through a batch it'd stall or restart, sometimes get through, sometimes not. At that point decided the honest move was to stop nickel-and-diming a free tier that clearly wasn't built for a multi-model pipeline like this, rather than keep shaving things down and risking accuracy to chase a $0 hosting bill. Tried Hugging Face Spaces next since it's built specifically for ML apps and generally has more headroom, but their free CPU tier turned out to require a Pro subscription for anything beyond static (no-backend) pages a policy I hadn't accounted for going in. Ended up subscribing to HF Pro ($9/month) and deploying there as a Docker Space instead, which is a small, predictable cost against not fighting a memory ceiling any further.

Deploying to a Docker Space meant writing an actual `Dockerfile` (installs `ffmpeg`, installs the pinned requirements, runs Streamlit on port 7860 since that's what HF Spaces expects rather than Streamlit's usual 8501) and a `README.md` with the specific YAML frontmatter HF Spaces reads to know it's a Docker-based app. Pushed via the `huggingface_hub` API directly rather than git, which let me create the Space, upload the code, and set its three secrets (HF token, dashboard username/password) all without needing a browser round-trip each time.

First real request against the freshly deployed Space was slow a minute or two because the Docker image doesn't have the models baked in, so Whisper's weights (139MB) download fresh the first time something actually needs them, inside the running container. That's expected and one-time per container lifetime, not a recurring cost; once cached, subsequent files in the same session skip the download entirely.

## What's actually weak here, no sugarcoating

- Emotional tone is the soft spot. One out of three isn't a real accuracy number, it's a sample size of three, but it does tell me the current approach needs either more labeled data or a stronger model built specifically for call-center audio rather than the general-purpose ones I'm using now.
- The background noise *type* field (is it a TV, static, chatter) is a rough guess based on how "flat" the noise spectrum looks, not a real classifier. A proper audio-event-tagging model would do this better; didn't have time to wire one in.
- The diarization model depends on a Hugging Face account and clicking through some gated-access pages an operational hassle for whoever sets this up in production, not a cost issue, but worth knowing about upfront.
- Every threshold I tuned (noise cutoffs, silence duration, overlap sensitivity) was tuned against three examples. That's enough to catch obviously broken logic, which it did, more than once, but it's not enough to trust the exact numbers blindly in production.
- Whisper occasionally garbles transcripts on quiet or non-English audio saw this directly on one of the three calls, which is partly in Spanish. That's an annoying correlation: the calls where audio quality is already rough are exactly the ones most likely to also break transcription, compounding the problem right when you need it to work most.

None of that is meant as a disclaimer to cover myself it's genuinely where I'd point someone with more time and more labeled data to look first.
