"""Refresh PyPI SHA-256 hashes for every exact pin in requirements.txt.

The file intentionally hashes every non-yanked artifact published for a pinned
release. That keeps the lock usable across supported platforms while making pip
verify the downloaded wheel or sdist. Release metadata is read only from PyPI's
JSON API and the replacement is atomic.
"""
from __future__ import annotations

import json
import argparse
import os
import pathlib
import re
import tempfile
import urllib.parse
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.txt"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)$")


def _hashes(name: str, version: str) -> list[str]:
    url = "https://pypi.org/pypi/{}/{}/json".format(
        urllib.parse.quote(name, safe=""), urllib.parse.quote(version, safe="")
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    files = payload.get("urls") or []
    hashes = sorted({
        str((item.get("digests") or {}).get("sha256") or "")
        for item in files if not item.get("yanked", False)
    } - {""})
    if not hashes:
        raise RuntimeError(f"{name}=={version} has no non-yanked PyPI artifacts")
    return hashes


def refresh(path: pathlib.Path = LOCK, only: set[str] | None = None) -> None:
    source = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    selected = {name.lower().replace("_", "-") for name in only} if only is not None else None
    index = 0
    while index < len(source):
        raw = source[index]
        match = PIN.fullmatch(raw.rstrip().removesuffix("\\").strip())
        if match is None:
            out.append(raw)
            index += 1
            continue
        name, version = match.groups()
        stop = index + 1
        while stop < len(source) and source[stop].lstrip().startswith("--hash=sha256:"):
            stop += 1
        if selected is not None and name.lower().replace("_", "-") not in selected:
            out.extend(source[index:stop])
            index = stop
            continue
        hashes = _hashes(name, version)
        out.append(f"{name}=={version} \\")
        for offset, digest in enumerate(hashes):
            suffix = " \\" if offset + 1 < len(hashes) else ""
            out.append(f"    --hash=sha256:{digest}{suffix}")
        index = stop

    payload = ("\n".join(out) + "\n").encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=".requirements.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.remove(temp_name)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=pathlib.Path, default=LOCK,
                        help="Exact-pinned lock to refresh (runtime lock by default)")
    parser.add_argument("--only", nargs="+", help="Refresh only these exact pinned packages")
    args = parser.parse_args()
    refresh(path=args.lock, only=set(args.only) if args.only is not None else None)
