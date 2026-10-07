"""Download the official Windows-x64 scanner only after SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import platform
import urllib.request
import zipfile

VERSION = "8.30.1"
ARCHIVE = f"gitleaks_{VERSION}_windows_x64.zip"
SHA256 = "d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e"
URL = f"https://github.com/gitleaks/gitleaks/releases/download/v{VERSION}/{ARCHIVE}"


def verified_executable(payload: bytes) -> bytes:
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("Scanner archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        matches = [item for item in archive.infolist() if item.filename == "gitleaks.exe"]
        if len(matches) != 1:
            raise ValueError("Scanner executable is missing or ambiguous")
        return archive.read(matches[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        parser.error("This pinned scanner is Windows x64 only")
    if args.output.exists():
        parser.error("Scanner output must not already exist")
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read(40 * 1024 * 1024 + 1)
    if len(payload) > 40 * 1024 * 1024:
        raise ValueError("Scanner download exceeds the size limit")
    executable = verified_executable(payload)
    args.output.mkdir(parents=True)
    (args.output / "gitleaks.exe").write_bytes(executable)
    print(f"Verified Gitleaks {VERSION}; archive SHA-256 {SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
