"""Grammar explanations for one sentence, via a local llama.cpp server or Claude.

EXPLAIN_BACKEND=local   any OpenAI-compatible server (llama.cpp's llama-server, Ollama, vLLM)
EXPLAIN_BACKEND=claude  Anthropic API; auth via ANTHROPIC_API_KEY or an `ant auth login` profile
EXPLAIN_BACKEND=off     hide the feature
"""

import json
import os
import re
import urllib.error
import urllib.request

BACKEND = os.environ.get("EXPLAIN_BACKEND", "local")
LANGUAGE = os.environ.get("EXPLAIN_LANGUAGE", "English")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "local")  # llama-server serves whatever model it loaded
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "low")  # short, simple task; raise if explanations feel shallow

SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "grammar": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "level": {"type": "string"},
                    "explanation": {"type": "string"},
                    "in_this_sentence": {"type": "string"},
                },
                "required": ["pattern", "level", "explanation", "in_this_sentence"],
                "additionalProperties": False,
            },
        },
        "vocab": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "reading": {"type": "string"},
                    "meaning": {"type": "string"},
                },
                "required": ["word", "reading", "meaning"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["translation", "grammar", "vocab", "notes"],
    "additionalProperties": False,
}

SYSTEM = f"""You are a Japanese teacher helping a JLPT N4-level learner who is shadowing listening-practice audio sentence by sentence.

For the target sentence, return:
- translation: a natural {LANGUAGE} translation.
- grammar: each grammar pattern, conjugation or set expression a learner at this level should notice (e.g. 〜てしまう, 〜なければならない, 〜んです, casual contractions). Give the pattern in its dictionary shape, its JLPT level ("N5".."N1", or "-" if not a JLPT item), a short explanation, and how it works in this exact sentence. Skip trivial particles unless their use here is notable.
- vocab: only words whose meaning in this context is not obvious from a dictionary (idioms, set phrases, words used in an unusual sense). Can be empty.
- notes: nuance, politeness level, anything a listener should pay attention to (e.g. sounds that get reduced in fast speech). If the sentence looks like a speech-recognition error, say so here.

Write explanations in {LANGUAGE}, concise and concrete. The text comes from automatic speech recognition and may contain mistakes."""


class ExplainError(Exception):
    pass


def model_name() -> str | None:
    return {"local": LLM_MODEL, "claude": CLAUDE_MODEL}.get(BACKEND)


def _prompt(sentence: str, before: list[str], after: list[str]) -> str:
    ctx = "\n".join(before + ["→ " + sentence] + after)
    return f"Context (target sentence marked with →):\n{ctx}\n\nTarget sentence:\n{sentence}"


def _local(user: str) -> dict:
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0.3,
        # llama-server turns the schema into a grammar, so the output is always valid JSON.
        "response_format": {"type": "json_schema", "json_schema": {"name": "explanation", "schema": SCHEMA, "strict": True}},
        # Qwen-style hybrid models: skip the thinking phase, it's slow and unnecessary here (needs --jinja).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    req = urllib.request.Request(LLM_BASE_URL.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        raise ExplainError(f"Local model returned {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    except urllib.error.URLError as e:
        raise ExplainError(f"Can't reach the local model at {LLM_BASE_URL} ({e}). Is llama-server running?") from e
    content = resp["choices"][0]["message"]["content"] or ""
    content = re.sub(r"^\s*<think>.*?</think>", "", content, flags=re.S)  # in case thinking wasn't disabled
    return json.loads(content)


def _claude(user: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY, or the `ant auth login` profile when that's unset
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if "haiku" not in CLAUDE_MODEL:  # Haiku 4.5 doesn't accept effort
        output_config["effort"] = CLAUDE_EFFORT
    kwargs = {"output_config": output_config}
    extra = {}
    if CLAUDE_MODEL.startswith(("claude-opus-5", "claude-fable-5")):
        # If a safety classifier declines, the API retries on a fallback model instead of failing.
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        extra["fallbacks"] = "default"
    try:
        resp = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            extra_body=extra or None,
            **kwargs,
        )
    except anthropic.AuthenticationError as e:
        raise ExplainError("Claude auth failed: set ANTHROPIC_API_KEY or run `ant auth login`.") from e
    except anthropic.APIConnectionError as e:
        raise ExplainError(f"Can't reach the Claude API: {e}") from e
    except anthropic.APIStatusError as e:
        raise ExplainError(f"Claude API error {e.status_code}: {e.message}") from e
    if resp.stop_reason == "refusal":
        raise ExplainError("Claude declined to explain this sentence.")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise ExplainError(f"No answer from Claude (stop_reason={resp.stop_reason}).")
    return json.loads(text)


def explain(sentence: str, before: list[str], after: list[str]) -> dict:
    user = _prompt(sentence, before, after)
    if BACKEND == "local":
        return _local(user)
    if BACKEND == "claude":
        return _claude(user)
    raise ExplainError("Explanations are turned off (EXPLAIN_BACKEND=off).")
