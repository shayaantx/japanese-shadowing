# Shadowing

Tiny local service: paste a YouTube URL (e.g. JLPT N4 listening practice) or open a video/audio file from disk, it transcribes the Japanese audio with Whisper, then plays it back **one sentence at a time** so you can shadow each line.

## Run with Docker

```sh
docker compose up --build                                          # CPU, model "small"
docker compose -f compose.yaml -f compose.gpu.yaml up --build      # NVIDIA GPU, model "large-v3-turbo"
```

Open http://127.0.0.1:8000. Transcripts and audio go to `./data`; Whisper models are cached in the `models` volume. Override the model with e.g. `WHISPER_MODEL=medium docker compose up`.

YouTube changes often and old yt-dlp versions stop working. If downloads start failing, bump it and rebuild: `uv lock --upgrade-package yt-dlp && docker compose build`.

## Run without Docker

Requires [uv](https://docs.astral.sh/uv/) (it fetches a compatible Python itself).

```sh
uv sync
uv run app.py            # http://127.0.0.1:8000
```

Recommended: have `ffmpeg` on PATH. It isn't strictly required, but without it yt-dlp saves YouTube's DASH m4a as-is, and some browsers seek less reliably in that container.

### Configuration (env vars)

| Var | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `small` | `large-v3-turbo` is much better for Japanese; use it if you have a GPU (or patience on CPU). `tiny`/`base` for quick tests. |
| `WHISPER_DEVICE` | `auto` | `cuda` / `cpu` |
| `WHISPER_COMPUTE` | `auto` | int8 on CPU, float16 on GPU |
| `HOST` | `127.0.0.1` | `0.0.0.0` to use it from your phone on the LAN |
| `PORT` | `8000` | |

On a GPU box: `WHISPER_MODEL=large-v3-turbo uv run app.py`.

## Player

- `Space` play / replay current sentence (stops at the end so you can shadow)
- `←` / `→` (or `Enter`) previous / next sentence
- `↑` / `↓` playback speed (pitch preserved)
- `T` toggle text (blur it for pure listening)
- `Esc` stop
- **Repeat N**: play each sentence N times with a shadowing pause in between
- **Auto-advance**: hands-free; after each sentence waits `sentence length × pause factor`, then moves on

Click any line in the list to jump. Speed/settings and your position per video are remembered.

## How it works

1. `yt-dlp` downloads the audio stream into `data/<video_id>/`. Or: a local file is uploaded to `data/file-<hash>/` and, if ffmpeg is available, its audio track is extracted to AAC (the video itself isn't kept). Without ffmpeg the file is played as-is, which works for browser-friendly formats (mp4, webm, mp3, m4a) but not e.g. mkv/avi.
2. `faster-whisper` transcribes with `language="ja"`, VAD filtering and word-level timestamps. A punctuated Japanese `initial_prompt` nudges Whisper into emitting `。`/`？`.
3. Words are regrouped into sentences: split on `。！？`, on pauses > 0.8 s, or on a small pause once a sentence runs over 12 s.
4. Result is cached in `data/<video_id>/transcript.json`, so the same URL is instant the second time (delete the folder to re-transcribe).
# japanese-shadowing
