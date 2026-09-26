"""Offline sentence breakdown: furigana + word list with JMdict glosses.

Tokenizing uses fugashi + unidic-lite (bundled dictionary, no setup).
English meanings come from JMdict via the jmdict-simplified JSON release, which
you download yourself and drop in data/dict/ (see README). On first start it's
converted to a small SQLite index so lookups are instant and memory stays low.
"""

import json
import os
import re
import sqlite3
import threading
import traceback
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "data"
DICT_DIR = DATA / "dict"
DB_PATH = DICT_DIR / "jmdict.sqlite"

KANJI = re.compile(r"[㐀-鿿豈-﫿々〆ヵヶ]")
KANA = re.compile(r"^[぀-ヿー]$")

# UniDic top-level part of speech -> label. Anything not listed (particles,
# auxiliaries, punctuation, suffixes) is left out of the word list.
POS = {
    "名詞": "noun",
    "代名詞": "pronoun",
    "動詞": "verb",
    "形容詞": "i-adjective",
    "形状詞": "na-adjective",
    "副詞": "adverb",
    "連体詞": "pre-noun",
    "接続詞": "conjunction",
    "感動詞": "interjection",
}

_tagger = None
_tagger_lock = threading.Lock()
dict_status = "not loaded"


def to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _feat(tok, name):
    v = getattr(tok.feature, name, None)
    return None if v in (None, "*", "") else v


def tagger():
    global _tagger
    with _tagger_lock:
        if _tagger is None:
            from fugashi import Tagger

            _tagger = Tagger()  # picks up unidic-lite automatically
        return _tagger


# ---------- JMdict index ----------

def _find_source() -> Path | None:
    env = os.environ.get("JMDICT_PATH")
    if env:
        return Path(env)
    found = sorted(DICT_DIR.glob("jmdict-eng*.json*"))
    return found[-1] if found else None


def build_index():
    """Convert jmdict-simplified JSON into SQLite once. Runs in a background thread."""
    global dict_status
    try:
        if DB_PATH.exists():
            dict_status = "ready"
            return
        src = _find_source()
        if not src or not src.exists():
            dict_status = "missing"
            return
        dict_status = "building"
        DICT_DIR.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                name = next(n for n in z.namelist() if n.endswith(".json"))
                jm = json.loads(z.read(name))
        else:
            jm = json.loads(src.read_text(encoding="utf-8"))

        tmp = DB_PATH.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        db = sqlite3.connect(tmp)
        db.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY, common INTEGER, data TEXT)")
        db.execute("CREATE TABLE forms (form TEXT, id INTEGER)")
        for w in jm["words"]:
            eid = int(w["id"])
            kanji = [k["text"] for k in w.get("kanji", [])]
            kana = [k["text"] for k in w.get("kana", [])]
            common = any(x.get("common") for x in w.get("kanji", []) + w.get("kana", []))
            senses = [
                [g["text"] for g in s.get("gloss", []) if g.get("lang", "eng") == "eng"]
                for s in w.get("sense", [])
            ]
            data = {"k": kanji, "r": kana, "s": [g for g in senses if g]}
            db.execute("INSERT INTO entries VALUES (?, ?, ?)", (eid, int(common), json.dumps(data, ensure_ascii=False)))
            db.executemany("INSERT INTO forms VALUES (?, ?)", [(f, eid) for f in set(kanji + kana)])
        db.execute("CREATE INDEX forms_form ON forms(form)")
        db.commit()
        db.close()
        tmp.replace(DB_PATH)
        dict_status = "ready"
    except Exception as e:
        traceback.print_exc()
        dict_status = f"error: {e}"


def lookup(forms: list[str], reading: str | None) -> dict | None:
    if dict_status != "ready":
        return None
    db = sqlite3.connect(DB_PATH)
    try:
        for form in dict.fromkeys(f for f in forms if f):
            rows = db.execute(
                "SELECT e.common, e.data FROM forms f JOIN entries e ON e.id = f.id WHERE f.form = ? LIMIT 20",
                (form,),
            ).fetchall()
            if not rows:
                continue
            # Prefer entries whose reading matches what was actually said (方: かた vs ほう), then common words.
            best = max(rows, key=lambda r: (reading in json.loads(r[1])["r"] if reading else False, r[0]))
            d = json.loads(best[1])
            return {"headword": d["k"][0] if d["k"] else d["r"][0], "senses": [s[:3] for s in d["s"][:3]]}
    finally:
        db.close()
    return None


# ---------- sentence analysis ----------

def _ruby(surface: str, reading: str | None) -> list[list]:
    """Split a token into [text, ruby|None] segments, keeping okurigana outside the ruby."""
    if not reading or not KANJI.search(surface):
        return [[surface, None]]
    s, r = surface, to_hira(reading)
    head = tail = ""
    while len(s) > 1 and r and KANA.match(s[-1]) and to_hira(s[-1]) == r[-1]:
        tail = s[-1] + tail
        s, r = s[:-1], r[:-1]
    while len(s) > 1 and r and KANA.match(s[0]) and to_hira(s[0]) == r[0]:
        head += s[0]
        s, r = s[1:], r[1:]
    segs = [[head, None]] if head else []
    segs.append([s, r or None])
    if tail:
        segs.append([tail, None])
    return segs


def analyze(text: str) -> dict:
    furigana, words, seen = [], [], set()
    for tok in tagger()(text):
        surface = tok.surface
        # Keep the whitespace MeCab strips before tokens (matters for mixed Latin text).
        if tok.white_space:
            furigana.append([tok.white_space, None])
        # "kana" is the written-style reading (センセイ); fall back to pronunciation (センセー).
        furigana.extend(_ruby(surface, _feat(tok, "kana") or _feat(tok, "pron")))

        pos1 = _feat(tok, "pos1")
        if pos1 not in POS or _feat(tok, "pos2") == "数詞":
            continue
        lemma = (_feat(tok, "lemma") or "").split("-")[0]  # UniDic writes loanwords as "コーヒー-coffee"
        base = _feat(tok, "orthBase") or lemma or surface
        if base in seen:
            continue
        seen.add(base)
        reading = to_hira(_feat(tok, "kanaBase") or _feat(tok, "pronBase") or "") or None
        words.append({
            "surface": surface,
            "base": base,
            "reading": reading if reading and reading != base else None,
            "pos": POS[pos1],
            "dict": lookup([base, lemma, surface], reading),
        })
    return {"furigana": furigana, "words": words, "dictionary": dict_status}
