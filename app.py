"""AutoAce voice-tone / background-noise dashboard.

Login -> upload a ZIP (audio files + labels.csv-style manifest at the
root) -> batch-process -> review results -> download CSV/JSON.
"""
import csv
import io
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud has no .env file -- secrets come from st.secrets instead.
# Mirror them into os.environ so pipeline.py/overlap.py (which just call
# os.getenv) don't need to know which environment they're running in.
if hasattr(st, "secrets"):
    for key in ("HF_TOKEN", "APP_USER", "APP_PASS"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]

import pipeline


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

st.set_page_config(page_title="AutoAce Voice Tone Dashboard", layout="wide")

SUPPORTED_EXTS = {".ogg", ".wav", ".mp3", ".m4a", ".flac"}
APP_USER = os.getenv("APP_USER", "autoace")
APP_PASS = os.getenv("APP_PASS", "changeme")


def check_login():
    if st.session_state.get("authed"):
        return True

    st.title("AutoAce Voice Tone Dashboard")
    with st.form("login"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if user == APP_USER and pw == APP_PASS:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    return False


def parse_manifest(zf, names):
    csv_names = [n for n in names if n.lower().endswith(".csv")]
    if not csv_names:
        return {}, "No CSV manifest found in the uploaded archive."
    manifest = {}
    with zf.open(csv_names[0]) as f:
        text = io.TextIOWrapper(f, encoding="utf-8")
        reader = csv.DictReader(text)
        if "name" not in (reader.fieldnames or []):
            return {}, f"Manifest {csv_names[0]} is missing a 'name' column."
        for row in reader:
            manifest[row["name"]] = row.get("result_json", "")
    return manifest, None


def run_batch(zf, names, manifest, work_dir):
    audio_names = [n for n in names if Path(n).suffix.lower() in SUPPORTED_EXTS]

    missing_from_manifest = [n for n in audio_names if Path(n).name not in manifest]
    missing_audio = [n for n in manifest if n not in [Path(a).name for a in audio_names]]

    results = []
    progress = st.progress(0, text="Starting...")
    total = len(audio_names)

    for i, name in enumerate(audio_names):
        fname = Path(name).name
        progress.progress((i) / max(total, 1), text=f"Processing {fname} ({i + 1}/{total})")
        out_path = Path(work_dir) / fname
        try:
            with zf.open(name) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            prediction, elapsed = pipeline.process(str(out_path))
            results.append({"name": fname, "status": "ok", "latency_sec": round(elapsed, 2), **prediction})
        except Exception as e:
            results.append({"name": fname, "status": "error", "error": str(e)})

    progress.progress(1.0, text="Done")
    return results, missing_from_manifest, missing_audio


def main():
    if not check_login():
        return

    st.title("AutoAce Voice Tone Dashboard")
    st.caption("Upload a ZIP containing audio files + a CSV manifest (name, result_json columns) at the root.")

    with st.sidebar:
        if st.button("Log out"):
            st.session_state["authed"] = False
            st.rerun()

    uploaded = st.file_uploader("Upload evaluation batch (.zip)", type=["zip"])
    if uploaded is None:
        st.info("Waiting for a batch upload.")
        return

    with zipfile.ZipFile(uploaded) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        manifest, err = parse_manifest(zf, names)
        if err:
            st.warning(f"{err} Proceeding without expected-value comparison.")

        audio_count = len([n for n in names if Path(n).suffix.lower() in SUPPORTED_EXTS])
        st.write(f"Found **{audio_count}** audio file(s), manifest has **{len(manifest)}** row(s).")

        if st.button("Run batch analysis", type="primary"):
            with tempfile.TemporaryDirectory() as work_dir:
                start = time.time()
                results, missing_from_manifest, missing_audio = run_batch(zf, names, manifest, work_dir)
                total_elapsed = time.time() - start

            if missing_from_manifest:
                st.warning(f"{len(missing_from_manifest)} audio file(s) have no matching manifest row: {missing_from_manifest}")
            if missing_audio:
                st.warning(f"{len(missing_audio)} manifest row(s) have no matching audio file: {missing_audio}")

            df = pd.DataFrame(results)
            ok_count = (df["status"] == "ok").sum() if "status" in df else 0
            st.success(f"Processed {ok_count}/{len(results)} files in {total_elapsed:.1f}s "
                       f"({total_elapsed / max(len(results), 1):.2f}s/file avg)")

            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(results, indent=2, cls=NpEncoder).encode("utf-8")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download results (CSV)", csv_bytes, "results.csv", "text/csv")
            with col2:
                st.download_button("Download results (JSON)", json_bytes, "results.json", "application/json")


if __name__ == "__main__":
    main()
