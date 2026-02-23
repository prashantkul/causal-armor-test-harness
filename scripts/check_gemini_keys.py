#!/usr/bin/env python3
"""Check which GCP project each Gemini API key in .env belongs to.

Calls the Gemini API with each key and inspects the response headers
and error details to identify the associated project.

Usage:
    python scripts/check_gemini_keys.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def extract_keys(env_path: Path) -> list[tuple[str, str]]:
    """Extract all GOOGLE_API_KEY values from .env (including commented-out)."""
    keys: list[tuple[str, str]] = []
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Match both active and commented-out keys
            m = re.match(r"^#?\s*GOOGLE_API_KEY\s*=\s*(.+)$", line)
            if m:
                key = m.group(1).strip().strip("'\"")
                status = "commented" if line.startswith("#") else "active"
                keys.append((key, status))
    return keys


def check_key(api_key: str) -> dict[str, str]:
    """Query the Gemini API to identify the project behind a key."""
    # Use the models.list endpoint — lightweight, no quota cost
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    info: dict[str, str] = {}

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            info["status"] = "valid"
            # Check for project info in response headers
            for header in ["x-goog-project-id", "x-goog-project-number"]:
                val = resp.headers.get(header)
                if val:
                    info[header] = val
    except urllib.error.HTTPError as e:
        info["status"] = f"error ({e.code})"
        try:
            body = json.loads(e.read().decode())
            error = body.get("error", {})
            info["message"] = error.get("message", "")[:200]
            # Extract project info from error details
            for detail in error.get("details", []):
                for violation in detail.get("violations", []):
                    metric = violation.get("quotaMetric", "")
                    if metric:
                        info["quota_metric"] = metric
        except Exception:
            pass
    except Exception as e:
        info["status"] = f"error: {e}"

    # Also try generateContent to trigger quota error which may reveal project
    gen_url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = json.dumps(
        {"contents": [{"parts": [{"text": "Hi"}]}]}
    ).encode()
    try:
        req = urllib.request.Request(
            gen_url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            info["generate"] = "ok"
    except urllib.error.HTTPError as e:
        info["generate"] = f"error ({e.code})"
        try:
            body = json.loads(e.read().decode())
            msg = body.get("error", {}).get("message", "")
            info["generate_message"] = msg[:300]
        except Exception:
            pass
    except Exception as e:
        info["generate"] = f"error: {e}"

    return info


def mask_key(key: str) -> str:
    """Show first 8 and last 4 chars."""
    if len(key) <= 12:
        return key
    return f"{key[:8]}...{key[-4:]}"


def main() -> None:
    keys = extract_keys(ENV_FILE)
    if not keys:
        print("No GOOGLE_API_KEY entries found in .env")
        return

    print(f"Found {len(keys)} Gemini API key(s) in .env\n")

    for i, (key, status) in enumerate(keys, 1):
        print(f"Key {i}: {mask_key(key)}  ({status})")
        info = check_key(key)

        print(f"  List models: {info.get('status', 'unknown')}")
        if "x-goog-project-id" in info:
            print(f"  Project ID: {info['x-goog-project-id']}")
        if "x-goog-project-number" in info:
            print(f"  Project Number: {info['x-goog-project-number']}")

        gen_status = info.get("generate", "")
        print(f"  Generate: {gen_status}")
        if "generate_message" in info:
            print(f"  Message: {info['generate_message']}")

        if info.get("status") == "valid" and "x-goog-project-id" not in info:
            print("  (project ID not in response headers)")

        print()


if __name__ == "__main__":
    main()
