"""Load model-provider credentials from this installation's ``.env``.

Real providers read their keys straight from ``os.environ``
(``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``DEEPSEEK_API_KEY``, ...). This is a
tiny, dependency-free ``.env`` reader (no python-dotenv). It populates
``os.environ`` *without* clobbering variables already set in the real shell
environment, so an explicit ``export OPENAI_API_KEY=...`` always wins.
``providers.get_provider`` calls :func:`load_provider_env` before building any
real provider.

WHERE the file lives depends on how the harness was installed
(:func:`resolve_env_path`):

  * **source checkout / editable install** --- ``harness/_vendor/ENV_MGMT/.env``,
    the historic location, git-ignored, documented by the committed
    ``.env.template`` next to it. This is the working layout, so it is preserved.
  * **installed wheel** --- a per-user config file
    (``%APPDATA%\\zora-harness\\.env``, or ``$XDG_CONFIG_HOME/zora-harness/.env``
    falling back to ``~/.config``). The package directory then lives inside
    site-packages, where writing a credential is wrong twice over: ``pip install
    -U`` silently deletes it, and a system-wide Python raises ``PermissionError``
    instead.
"""
from __future__ import annotations

import os
import pathlib
import re

# Directory name used for the per-user config location (matches the distribution
# name in pyproject.toml).
APP_DIR = "zora-harness"

# Historic in-repo location: still the read/write target for a source checkout.
VENDORED_ENV_PATH = (pathlib.Path(__file__).resolve().parent
                     / "_vendor" / "ENV_MGMT" / ".env")


def is_source_checkout() -> bool:
    """True when the package is being used from the repo (or an editable install).

    Detected by ``pyproject.toml`` sitting next to the package directory --- it is
    present in the source tree and is never installed into site-packages, so this
    distinguishes "running from the checkout" from "running from a wheel" without
    guessing at path substrings.
    """
    return (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").is_file()


def user_env_path() -> pathlib.Path:
    """Per-user, writable ``.env`` location for an installed (non-source) harness.

    Windows uses ``%APPDATA%`` (roaming per-user config, owner-scoped by its
    inherited ACL); everything else follows the XDG base-directory spec.
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        root = (pathlib.Path(base) if base
                else pathlib.Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME")
        root = pathlib.Path(base) if base else pathlib.Path.home() / ".config"
    return root / APP_DIR / ".env"


def resolve_env_path() -> pathlib.Path:
    """The single ``.env`` this installation reads from and writes to."""
    return VENDORED_ENV_PATH if is_source_checkout() else user_env_path()


ENV_PATH = resolve_env_path()

_loaded = False


def _parse(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; skip comments/blanks; strip quotes.

    Blank right-hand sides are dropped so an unfilled template key behaves
    exactly like "not set" (the provider then raises a clean "<KEY> is not set").
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key and val:
            out[key] = val
    return out


def _write_owner_only(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` with owner-only permissions where supported.

    The mode is set AT CREATION (``os.open`` with 0o600) rather than chmod-ed
    afterwards, so a fresh key file is never briefly world-readable; the trailing
    ``chmod`` then tightens a file that already existed with looser bits.

    Windows has no POSIX mode bits --- ``os.chmod`` there only toggles the
    read-only attribute --- so on Windows the file's protection comes from the
    ACL it inherits from its directory instead (``%APPDATA%\\zora-harness`` and a
    user-owned checkout are both owner-scoped by default).
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    try:
        os.chmod(path, 0o600)
    except OSError:                       # pragma: no cover - exotic filesystem
        pass


def save_key(name: str, value: str) -> pathlib.Path:
    """Persist ``name=value`` into this installation's ``.env`` *and* set it in
    ``os.environ`` for the current process.

    The file is :data:`ENV_PATH` --- ``harness/_vendor/ENV_MGMT/.env`` in a source
    checkout (git-ignored), otherwise the per-user config path from
    :func:`user_env_path`; see the module docstring for why an installed harness
    must not write into site-packages. It is created with mode ``0o600``
    (owner read/write only) on POSIX; on Windows, which has no mode bits, it
    relies on the owner-scoped ACL of its parent directory.

    Setting the process env directly is what makes a freshly-entered key take
    effect immediately: :func:`load_provider_env` is memoised (its ``_loaded``
    guard), so a mid-session ``.env`` edit alone would not be re-read until a
    restart. An existing (uncommented) line for ``name`` is replaced in place;
    otherwise the pair is appended. The parent dir/file are created if absent.

    Secret hygiene: the value is written to that file and the process env only
    --- it is never returned or logged. Raises ``ValueError`` on an empty name or
    value.
    """
    name = (name or "").strip()
    value = (value or "").strip()
    if not name:
        raise ValueError("env var name is required")
    if not value:
        raise ValueError("value is empty (paste the key first)")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid env var name {name!r}")
    if any(ch in value for ch in ("\r", "\n", "\0")):
        raise ValueError("env value must be one line and contain no NUL byte")

    # 0o700 so a freshly created per-user config dir is not listable by others
    # (no-op on Windows, and masked by umask on POSIX, which is fine).
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    old_lines = (ENV_PATH.read_text(encoding="utf-8").splitlines()
                 if ENV_PATH.exists() else [])
    new_line = f"{name}={value}"

    def _key_of(raw: str) -> str | None:
        s = raw.strip()
        if not s or s.startswith("#"):
            return None
        if s.startswith("export "):
            s = s[len("export "):].lstrip()
        return s.partition("=")[0].strip() if "=" in s else None

    # Replace the first matching line in place and DROP any later duplicate, so the
    # file ends with exactly one authoritative line for ``name``. _parse (the reader)
    # is last-wins, so leaving a stale trailing duplicate would silently revert the
    # just-saved key on the next launch.
    kept: list[str] = []
    replaced = False
    for raw in old_lines:
        if _key_of(raw) == name:
            if not replaced:
                kept.append(new_line)
                replaced = True
            # else: swallow the duplicate
        else:
            kept.append(raw)
    if not replaced:
        kept.append(new_line)
    _write_owner_only(ENV_PATH, "\n".join(kept) + "\n")

    os.environ[name] = value          # effective now, bypasses the _loaded memo
    return ENV_PATH


def load_provider_env(*, override: bool = False) -> dict[str, str]:
    """Populate ``os.environ`` from :data:`ENV_PATH`; return the keys applied.

    Idempotent: after the first successful pass it is a no-op unless
    ``override=True``. Existing environment variables are kept unless
    ``override`` is set. A missing ``.env`` file is a silent no-op.
    """
    global _loaded
    applied: dict[str, str] = {}
    if _loaded and not override:
        return applied
    if ENV_PATH.exists():
        for key, val in _parse(ENV_PATH.read_text(encoding="utf-8")).items():
            if override or key not in os.environ:
                os.environ[key] = val
                applied[key] = val
    _loaded = True
    return applied
