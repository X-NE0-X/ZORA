"""Fail-closed local distribution checks; never publish or modify Git history."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md", "LICENSE", "NOTICE",
    "provenance.json", "CONTRIBUTING.md", "RELEASE.md", "pyproject.toml",
    "MANIFEST.in", "requirements.txt", "requirements-release.txt",
    "run_config.json", "field_config.json", ".gitignore",
    ".github/workflows/ci.yml", "data/README.md", "data/field_ingest.py",
)
PRIVATE_PARTS = {".git", ".venv", "venv", "env", "artifacts", "memory",
                 "ArchiveDeck", "FLEET_SCIENCE", "__pycache__", ".pytest_cache",
                 ".codex", ".agents", ".claude", ".vscode", ".aws", ".ssh", ".idea"}
TEMPLATE = "harness/_vendor/ENV_MGMT/.env.template"
LICENSE_EXPRESSION = "LicenseRef-ZORA-Source-Available"


class ReleaseError(ValueError):
    """An invalid or incomplete local release candidate."""


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute() \
            or ".." in path.parts or str(path) != name:
        raise ReleaseError("Unsafe archive path")
    if any(part in PRIVATE_PARTS for part in path.parts):
        raise ReleaseError(f"Private runtime state in archive: {name}")
    if path.suffix.lower() in {".parquet", ".pyc", ".pyo", ".pem", ".key"}:
        raise ReleaseError(f"Private/binary file in archive: {name}")
    if path.name.startswith(".env") and name != TEMPLATE:
        raise ReleaseError(f"Credential file in archive: {name}")
    return name


def _regular(path: Path, root: Path) -> None:
    for item in (path, *path.parents):
        if item == root.parent:
            break
        if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
            raise ReleaseError("Release sources must not contain links or junctions")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ReleaseError("Missing or escaping release source")


def public_files(root: Path) -> dict[str, bytes]:
    """Explicit source selection, including untracked current implementation."""
    paths = [root / name for name in ROOT_FILES]
    for directory in ("harness", "tests", "tools"):
        paths.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".json"}
                     and not any(part in PRIVATE_PARTS for part in p.relative_to(root).parts))
    paths.append(root / TEMPLATE)
    result = {}
    for path in sorted(set(paths)):
        _regular(path, root)
        name = safe_name(path.relative_to(root).as_posix())
        result[name] = path.read_bytes()
    return result


def validate_source(root: Path, files: dict[str, bytes]) -> None:
    config = tomllib.loads(files["pyproject.toml"].decode("utf-8"))["project"]
    if config.get("license") != LICENSE_EXPRESSION:
        raise ReleaseError("Combined license must not be labeled Apache-2.0 alone")
    for field in ("authors", "maintainers"):
        if config.get(field) != [{"name": "X-NE0-X"}]:
            raise ReleaseError("Public metadata must use the approved handle only")
    if not {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "provenance.json"} \
            <= set(config.get("license-files", [])):
        raise ReleaseError("Required license assets are not declared")
    license_text = files["LICENSE"].decode("utf-8")
    if '"Commons Clause" License Condition v1.0' not in license_text \
            or "END OF TERMS AND CONDITIONS" not in license_text:
        raise ReleaseError("Incomplete combined license")
    provenance = json.loads(files["provenance.json"])
    expected = {f"harness/_vendor/{name}" for name in
                ("BacktestEngine", "CTX", "FactorEngine", "ENV_MGMT")}
    components = provenance.get("components", [])
    if {item.get("path") for item in components} != expected \
            or any(item.get("origin") != "first-party" or item.get("license_file") != "LICENSE"
                   for item in components):
        raise ReleaseError("Missing component provenance")
    for lock in ("requirements.txt", "requirements-release.txt"):
        groups = re.split(r"(?m)^(?=[A-Za-z0-9_.-]+==)", files[lock].decode("utf-8"))
        pins = [group for group in groups if re.match(r"[A-Za-z0-9_.-]+==", group)]
        if not pins or any("--hash=sha256:" not in group for group in pins):
            raise ReleaseError(f"Unhashed dependency pin in {lock}")
    # The public selection, unlike the live checkout, must not carry a machine
    # identity. Gitleaks is a separate gate; this check is not a secret scanner.
    for name, payload in files.items():
        if re.search(rb"[A-Za-z]:[/\\]Users[/\\](?!<|example|test|fake)[^/\\\s]+", payload):
            raise ReleaseError(f"Personal Windows profile path in {name}")


def archive_files(path: Path) -> dict[str, bytes]:
    result = {}
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ReleaseError("Wheel contains a symbolic link")
                name = safe_name(info.filename)
                if name in result:
                    raise ReleaseError("Duplicate wheel member")
                result[name] = archive.read(info)
    else:
        with tarfile.open(path, "r:gz") as archive:
            top = None
            for info in archive.getmembers():
                if info.isdir():
                    continue
                if not info.isfile():
                    raise ReleaseError("Source archive contains a link or special file")
                full = PurePosixPath(info.name)
                if full.is_absolute() or ".." in full.parts or len(full.parts) < 2:
                    raise ReleaseError("Unsafe source archive member")
                if top is None:
                    top = full.parts[0]
                if full.parts[0] != top:
                    raise ReleaseError("Source archive has multiple roots")
                name = safe_name(PurePosixPath(*full.parts[1:]).as_posix())
                if name in result:
                    raise ReleaseError("Duplicate source archive member")
                handle = archive.extractfile(info)
                if handle is None:
                    raise ReleaseError("Unreadable source archive member")
                result[name] = handle.read()
    return result


def verify_archive(path: Path, source: dict[str, bytes]) -> dict:
    files = archive_files(path)
    wheel = path.suffix == ".whl"
    required = {name: value for name, value in source.items()
                if not wheel or name.startswith("harness/")}
    if not wheel:
        required.pop(".gitignore")  # setuptools does not include ignore policy
    for name, payload in required.items():
        if files.get(name) != payload:
            raise ReleaseError(f"Missing or stale release member: {name}")
    if wheel:
        generated = {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt"}
        for name in files.keys() - required.keys():
            parts = PurePosixPath(name).parts
            if not (parts[0].endswith(".dist-info") and
                    (len(parts) == 2 and parts[1] in generated or
                     len(parts) == 3 and parts[1] == "licenses" and parts[2] in
                     {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "provenance.json"})):
                raise ReleaseError(f"Unapproved wheel member: {name}")
    else:
        generated = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
                     "requires.txt", "top_level.txt"}
        for name in files.keys() - required.keys():
            parts = PurePosixPath(name).parts
            if not (name in {"PKG-INFO", "setup.cfg"} or
                    len(parts) == 2 and parts[0].endswith(".egg-info") and parts[1] in generated):
                raise ReleaseError(f"Unapproved source archive member: {name}")
    legal = {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "provenance.json"}
    if wheel:
        for name in legal:
            matches = [value for key, value in files.items()
                       if key.endswith(".dist-info/licenses/" + name)]
            if matches != [source[name]]:
                raise ReleaseError(f"Wheel is missing the complete legal asset: {name}")
        metadata = [value for key, value in files.items() if key.endswith(".dist-info/METADATA")]
    else:
        metadata = [files.get("PKG-INFO", b"")]
    if len(metadata) != 1:
        raise ReleaseError("Missing or ambiguous package metadata")
    parsed = BytesParser().parsebytes(metadata[0])
    if parsed.get("License-Expression") != LICENSE_EXPRESSION:
        raise ReleaseError("Distribution license metadata differs from the source")
    if set(parsed.get_all("License-File", [])) != legal:
        raise ReleaseError("Distribution legal-file metadata is incomplete")
    return {"archive": path.name, "members": len(files),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "complete": True}


def export_snapshot(destination: Path, source: dict[str, bytes]) -> None:
    """Local-only opt-in export; refuses overwrite and never copies .git."""
    if destination.exists():
        raise ReleaseError("Snapshot destination must not already exist")
    destination.mkdir(parents=True)
    for name, payload in source.items():
        target = destination / safe_name(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--gitleaks", type=Path,
                        help="Locally scan the allowlisted source with a verified scanner")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Existing release evidence is never overwritten")
    try:
        source = public_files(args.source.resolve())
        validate_source(args.source, source)
        wheels, sdists = list(args.dist.glob("*.whl")), list(args.dist.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ReleaseError("Exactly one wheel and one sdist are required")
        receipts = [verify_archive(path, source) for path in wheels + sdists]
        source_scan = None
        if args.gitleaks:
            with tempfile.TemporaryDirectory(prefix="zora-release-scan-") as scratch:
                candidate = Path(scratch) / "source"
                export_snapshot(candidate, source)
                scan = subprocess.run([str(args.gitleaks.resolve()), "dir", str(candidate),
                                       "--redact=100", "--no-banner"],
                                      capture_output=True, text=True, encoding="utf-8", timeout=120)
            scan_output = scan.stdout + scan.stderr
            scanned_bytes = re.findall(r"scanned ~(\d+) bytes", scan_output)
            if scan.returncode != 0 or len(scanned_bytes) != 1 or int(scanned_bytes[0]) < 1 \
                    or re.search(r"\bERR\b|fatal:", scan_output):
                raise ReleaseError("Source secret scanner failed; run a redacted local scan for details")
            source_scan = {"complete": True, "scanned_bytes": int(scanned_bytes[0]), "redacted": True}
        if args.snapshot:
            export_snapshot(args.snapshot, source)
        manifest = {name: hashlib.sha256(payload).hexdigest() for name, payload in source.items()}
        payload = {"complete": True, "publication_authorized": False,
                   "source_files": manifest, "archives": receipts, "source_secret_scan": source_scan}
    except (ReleaseError, OSError, ValueError, tarfile.TarError, zipfile.BadZipFile,
            subprocess.SubprocessError) as error:
        payload = {"complete": False, "publication_authorized": False,
                   "error": str(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "source_files"}))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
