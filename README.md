# Shadowing

Tiny local service: paste a YouTube URL (e.g. JLPT N4 listening practice) or open a video/audio file from disk, it transcribes the Japanese audio with Whisper, then plays it back **one sentence at a time** so you can shadow each line.

## One-time setup

1. **Lock dependencies** (after cloning or changing `pyproject.toml`): `uv lock`. The Docker build uses `uv.lock` as-is.
2. **Dictionary (optional, for English word meanings):** download `jmdict-eng-<version>.json.zip` (the full English one, not `common`) from the [jmdict-simplified releases](https://github.com/scriptin/jmdict-simplified/releases) and put it in `data/dict/`. On first start it's indexed into `data/dict/jmdict.sqlite` (takes a minute, once). Without it you still get furigana and the word list, just no meanings.
3. **Grammar explanations (optional):** either
   - **Local:** run llama.cpp's server with a Japanese-capable model you've downloaded, e.g. Qwen3.8-27B on a 24 GB GPU or Qwen3.5-4B on 8 GB:
     ```sh
     llama-server -m Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -c 8192 --jinja --host 0.0.0.0 --port 8080
     ```
     `--jinja` lets the app switch off Qwen's thinking mode (much faster); `--host 0.0.0.0` lets the Docker container reach it. Ollama or vLLM work too: point `LLM_BASE_URL` at their OpenAI-compatible endpoint and set `LLM_MODEL`.
   - **Claude:** `EXPLAIN_BACKEND=claude`, and either set `ANTHROPIC_API_KEY` or sign in with your developer account via the [`ant` CLI](https://platform.claude.com/docs/en/cli-sdks-libraries/cli/quickstart) (`ant auth login`) and uncomment the `~/.config/anthropic` mount in `compose.yaml`. Make sure `ANTHROPIC_API_KEY` is *unset* when using the login; a set key (even empty) takes precedence.
   - Or `EXPLAIN_BACKEND=off` to hide the button.

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
| `EXPLAIN_BACKEND` | `local` | `local` (llama.cpp or any OpenAI-compatible server), `claude`, or `off` |
| `EXPLAIN_LANGUAGE` | `English` | Language of explanations, e.g. `simple Japanese` |
| `LLM_BASE_URL` | `http://localhost:8080/v1` | Docker compose defaults it to the host's llama-server |
| `LLM_MODEL` | `local` | Ignored by llama-server; needed for Ollama/vLLM |
| `LLM_API_KEY` | | If your server requires one |
| `CLAUDE_MODEL` | `claude-opus-5` | or `claude-sonnet-5` / `claude-haiku-4-5` for cheaper |
| `CLAUDE_EFFORT` | `low` | Thinking effort; raise to `medium`/`high` if explanations feel shallow |
| `JMDICT_PATH` | newest `data/dict/jmdict-eng*.json*` | Explicit path to the JMdict file |
| `HOST` | `127.0.0.1` | `0.0.0.0` to use it from your phone on the LAN |
| `PORT` | `8000` | |

On a GPU box: `WHISPER_MODEL=large-v3-turbo uv run app.py`.

## Player

- `Space` play / replay current sentence (stops at the end so you can shadow)
- `←` / `→` (or `Enter`) previous / next sentence
- `↑` / `↓` playback speed (pitch preserved)
- `T` toggle text (blur it for pure listening; also blurs the side panel)
- `F` toggle furigana on the main sentence
- `E` explain grammar for the current sentence
- `Esc` stop
- **Repeat N**: play each sentence N times with a shadowing pause in between
- **Auto-advance**: hands-free; after each sentence waits `sentence length × pause factor`, then moves on

Click any line in the list to jump. Speed/settings and your position per video are remembered.

### Side panel

- **Sentence** with furigana and **Words** (dictionary form, reading, part of speech, JMdict meanings): computed locally with fugashi + UniDic, free and instant.
- **Explain grammar**: translation, grammar patterns with JLPT level and how they're used in *this* sentence, context-specific vocab, nuance notes. The two sentences before and after are sent as context. Results are saved in `data/<id>/explanations.json`, so each sentence is only generated once (use **Regenerate** to redo it). The text comes from speech recognition, so the model is asked to flag likely transcription errors.

## How it works

1. `yt-dlp` downloads the audio stream into `data/<video_id>/`. Or: a local file is uploaded to `data/file-<hash>/` and, if ffmpeg is available, its audio track is extracted to AAC (the video itself isn't kept). Without ffmpeg the file is played as-is, which works for browser-friendly formats (mp4, webm, mp3, m4a) but not e.g. mkv/avi.
2. `faster-whisper` transcribes with `language="ja"`, VAD filtering and word-level timestamps. A punctuated Japanese `initial_prompt` nudges Whisper into emitting `。`/`？`.
3. Words are regrouped into sentences: split on `。！？`, on pauses > 0.8 s, or on a small pause once a sentence runs over 12 s.
4. Result is cached in `data/<video_id>/transcript.json`, so the same URL is instant the second time (delete the folder to re-transcribe).
# japanese-shadowing
