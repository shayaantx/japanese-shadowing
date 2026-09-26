"""Tiny local shadowing service.

YouTube URL or local video/audio file -> Japanese transcript (faster-whisper)
-> sentences with timestamps -> sentence-by-sentence player in the browser.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import traceback
import uuid
from pathlib import Path

import yt_dlp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# "small" is a reasonable default for a CPU box; "large-v3-turbo" is much more
# accurate for Japanese if you have a GPU or the patience.
MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "auto")

# Whisper tends to copy the style of the prompt, so a punctuated Japanese
# prompt nudges it into emitting 。 and ？, which we use to split sentences.
INITIAL_PROMPT = "こんにちは。今日は、日本語の聞き取りの練習をしましょう。準備はいいですか？"

SENTENCE_END = re.compile(r"[。．！？!?]$")
PAUSE_SPLIT = 0.8  # seconds of silence that ends a sentence even without punctuation
MAX_SENTENCE = 12.0  # beyond this, split at the next small pause

app = FastAPI()
jobs: dict[str, dict] = {}
_model = None
_model_lock = threading.Lock()
_work_lock = threading.Lock()  # one transcription at a time; they're CPU/GPU heavy


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        return _model


def split_sentences(segments) -> list[dict]:
    words = [w for seg in segments for w in (seg.words or [])]
    sentences, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = nxt.start - w.end if nxt else float("inf")
        dur = w.end - cur[0].start
        if SENTENCE_END.search(w.word.strip()) or gap > PAUSE_SPLIT or (dur > MAX_SENTENCE and gap > 0.25):
            text = "".join(x.word for x in cur).strip()
            if text:
                sentences.append({"start": round(cur[0].start, 2), "end": round(cur[-1].end, 2), "text": text})
            cur = []
    return sentences


def transcribe_into(job: dict, vdir: Path, audio: Path):
    """Transcribe `audio` and write vdir/transcript.json. Caller holds _work_lock."""
    job.update(status="loading model", progress=0.0)
    model = get_model()
    job.update(status="transcribing")
    segments, tinfo = model.transcribe(
        str(audio),
        language="ja",
        word_timestamps=True,
        vad_filter=True,
        initial_prompt=INITIAL_PROMPT,
    )
    done = []
    for seg in segments:  # generator: transcription happens as we iterate
        done.append(seg)
        job["progress"] = min(seg.end / tinfo.duration, 1.0) if tinfo.duration else 0
        job["preview"] = seg.text

    sentences = split_sentences(done)
    (vdir / "transcript.json").write_text(
        json.dumps(
            {"video_id": job["video_id"], "title": job["title"], "audio": audio.name, "model": MODEL_NAME, "sentences": sentences},
            ensure_ascii=False,
            indent=1,
        )
    )
    job.update(status="done", progress=1.0)


def run_url_job(job_id: str, url: str):
    job = jobs[job_id]
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        vid = info["id"]
        job.update(video_id=vid, title=info.get("title", vid))
        vdir = DATA / vid
        transcript = vdir / "transcript.json"
        if transcript.exists():
            job.update(status="done", progress=1.0)
            return

        job.update(status="queued")
        with _work_lock:
            if transcript.exists():  # same video finished by another job while we waited
                job.update(status="done", progress=1.0)
                return
            job.update(status="downloading", progress=0.0)

            def hook(d):
                if d["status"] == "downloading" and d.get("total_bytes"):
                    job["progress"] = d["downloaded_bytes"] / d["total_bytes"]

            opts = {
                # Plain audio streams: no ffmpeg needed for download or playback.
                "format": "bestaudio[ext=m4a]/bestaudio",
                "outtmpl": str(vdir / "audio.%(ext)s"),
                "noplaylist": True,
                "quiet": True,
                "progress_hooks": [hook],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                dl = ydl.extract_info(url, download=True)
            transcribe_into(job, vdir, Path(dl["requested_downloads"][0]["filepath"]))
    except Exception as e:
        traceback.print_exc()
        job.update(status="error", error=str(e))


def run_file_job(job_id: str, vdir: Path, source: Path):
    job = jobs[job_id]
    try:
        job.update(status="queued")
        with _work_lock:
            if (vdir / "transcript.json").exists():
                job.update(status="done", progress=1.0)
                return
            audio = source
            if shutil.which("ffmpeg"):
                # Pull out just the audio as AAC: any container becomes browser-playable
                # and seekable, and we don't keep a copy of the whole video around.
                job.update(status="extracting audio")
                audio = vdir / "audio.m4a"
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vn",
                     "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(audio)],
                    check=True,
                )
                source.unlink()
            transcribe_into(job, vdir, audio)
    except Exception as e:
        traceback.print_exc()
        job.update(status="error", error=str(e))


class JobIn(BaseModel):
    url: str


@app.post("/api/jobs")
def create_job(body: JobIn):
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"id": job_id, "status": "resolving", "progress": 0.0}
    threading.Thread(target=run_url_job, args=(job_id, body.url), daemon=True).start()
    return jobs[job_id]


@app.post("/api/upload")
async def upload(request: Request, name: str):
    # The browser sends the raw file as the request body (no multipart), streamed to disk.
    tmp_dir = DATA / ".uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp = tmp_dir / uuid.uuid4().hex
    sha = hashlib.sha1()
    with tmp.open("wb") as f:
        async for chunk in request.stream():
            sha.update(chunk)
            f.write(chunk)

    # Content-addressed id, so re-selecting the same file reuses its transcript.
    vid = "file-" + sha.hexdigest()[:12]
    vdir = DATA / vid
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "video_id": vid, "title": Path(name).stem}
    if (vdir / "transcript.json").exists():
        tmp.unlink()
        jobs[job_id].update(status="done", progress=1.0)
        return jobs[job_id]

    vdir.mkdir(exist_ok=True)
    source = vdir / ("source" + Path(name).suffix.lower())
    tmp.replace(source)
    threading.Thread(target=run_file_job, args=(job_id, vdir, source), daemon=True).start()
    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]


@app.get("/api/library")
def library():
    items = []
    for t in sorted(DATA.glob("*/transcript.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        d = json.loads(t.read_text())
        items.append({"video_id": d["video_id"], "title": d["title"], "count": len(d["sentences"])})
    return items


def _transcript(vid: str) -> dict:
    t = DATA / vid / "transcript.json"
    if "/" in vid or ".." in vid or not t.exists():
        raise HTTPException(404)
    return json.loads(t.read_text())


@app.get("/api/transcript/{vid}")
def transcript(vid: str):
    return _transcript(vid)


@app.get("/media/{vid}")
def media(vid: str):
    # FileResponse handles Range requests, which the <audio> element needs for seeking.
    return FileResponse(DATA / vid / _transcript(vid)["audio"])


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
