# CPU by default. For NVIDIA GPUs, build with a CUDA 12 + cuDNN 9 base instead
# (see compose.gpu.yaml); faster-whisper/ctranslate2 needs those libraries.
ARG BASE_IMAGE=debian:bookworm-slim
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

# ffmpeg lets yt-dlp remux YouTube's DASH audio into a normal, easily seekable file.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv fetches its own Python, so this works on both the Debian and CUDA base images.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock ./
# --locked: fail the build if uv.lock is out of date with pyproject.toml (run `uv lock`).
RUN uv sync --locked --no-install-project --python 3.12

COPY app.py analyze.py explain.py ./
COPY static ./static

ENV PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/cache/huggingface

EXPOSE 8000
VOLUME ["/app/data", "/cache"]
CMD ["python", "app.py"]
