"""Run one independent Cohere experiment and retain its original response."""
import argparse
from datetime import datetime, timezone
from getpass import getpass
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time

import cohere


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--json-mode", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("Назва запуску: лише латинські літери, цифри, _ та -")
    prompt = args.prompt.read_text(encoding="utf-8-sig")
    if not prompt.strip() or "{{" in prompt:
        parser.error("Промпт порожній або містить незаповнений шаблон {{...}}")
    api_key = os.getenv("COHERE_API_KEY") or getpass("Cohere API key (ввід приховано): ")
    if not api_key.strip():
        parser.error("API-ключ не введено")
    request = {
        "model": os.getenv("COHERE_MODEL", "command-a-03-2025"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    if args.json_mode:
        request["response_format"] = {"type": "json_object"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = Path("runs") / f"{stamp}_{args.name}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    metadata = {
        "created_at_utc": stamp,
        "cohere_sdk": version("cohere"),
        "python": sys.version.split()[0],
        "request": request,
    }
    client = cohere.ClientV2(api_key=api_key.strip(), timeout=90)
    started = time.perf_counter()
    try:
        response = client.chat(**request)
    except Exception as exc:
        # Do not save exception bodies, headers or authentication credentials.
        metadata.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        (folder / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Помилка {metadata['error_type']}, HTTP {metadata['status_code']}. Журнал: {folder}")
        return 1
    raw = response.model_dump(mode="json")
    text = "".join(
        block.text for block in (response.message.content or [])
        if block.type == "text"
    )
    metadata.update({
        "status": "received",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "response_id": response.id,
        "finish_reason": raw.get("finish_reason"),
        "usage": raw.get("usage"),
    })
    (folder / "response.txt").write_text(text, encoding="utf-8")
    (folder / "response_raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(text)
    print(f"\nЗбережено: {folder.resolve()}")
    finish = str(raw.get("finish_reason", "")).upper()
    if finish != "COMPLETE" or not text:
        print("Відповідь порожня або генерацію не завершено. Перевірте finish_reason.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
