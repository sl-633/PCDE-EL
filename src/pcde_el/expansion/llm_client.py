"""Ollama client and the resumable generation loop shared by dictionary and
query expansion."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from tqdm import tqdm

GENERATION_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 20,
    "repeat_penalty": 1.1,
    "num_predict": 4096,
}
# The reasoning trace is kept in the raw response, followed by this marker.
THINK_SEPARATOR = "</think>"


def default_think(model: str) -> str | bool:
    return "low" if "gpt-oss" in model else False


class OllamaGenerator:
    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        think: str | bool | None = None,
    ):
        from ollama import Client

        self.model = model
        self.client = Client(host=host)
        self.think = default_think(model) if think is None else think

    def __call__(self, prompt: str) -> tuple[str, float]:
        """Returns (raw response, seconds taken)."""
        t0 = time.time()
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            think=self.think,
            options=GENERATION_OPTIONS,
        )
        text = ""
        if getattr(response.message, "thinking", None):
            text += response.message.thinking + THINK_SEPARATOR
        if getattr(response.message, "content", None):
            text += response.message.content
        return text, time.time() - t0


def answer_part(raw: str) -> str:
    return raw.split(THINK_SEPARATOR)[-1].strip()


def clean_names(values) -> list[str]:
    """Collapse whitespace and drop empty and case-insensitive duplicate names.
    Anything other than a list counts as no names."""
    if not isinstance(values, list):
        return []
    out, seen = [], set()
    for v in values:
        s = " ".join(str(v).split())
        if s and s.casefold() not in seen:
            seen.add(s.casefold())
            out.append(s)
    return out


def load_json_or(path: Path, default: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json_atomic(obj: dict, path: Path) -> None:
    # Write to a temp file and rename, so an interrupted run never leaves a
    # truncated cache behind.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_resumable(
    keys: Iterable[str],
    generate: Callable[[str], dict],
    output: dict,
    json_path: Path,
    save_every: int = 100,
    limit: int = 0,
) -> int:
    """Call generate(key) for every key not yet in output["results"].

    Progress is saved to json_path every `save_every` keys, so a rerun picks
    up where the last one stopped. Returns the number of new results.
    """
    todo = [k for k in keys if k not in output["results"]]
    if limit:
        todo = todo[:limit]
    new = 0
    for key in tqdm(todo, desc="generating"):
        record = generate(key)
        output["results"][key] = record
        new += 1
        if not record.get("parse_success", True):
            tqdm.write(f"Parse failed: {key!r}")
        if new % save_every == 0:
            save_json_atomic(output, json_path)
    save_json_atomic(output, json_path)
    return new
