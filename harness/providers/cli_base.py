"""Shared helpers for CLI-backed providers (Codex, OpenCode).

These providers shell out to an agent CLI and read the answer off stdout, so the
harness can drive ChatGPT-login (Codex) or opencode without any SDK. The binary
is only required when the provider is actually used.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import signal
import time

from ..cancellation import check_cancelled

# Vendor credentials the harness may have loaded into the environment (from
# ENV_MGMT/.env or the shell). A third-party CLI spawned by the harness must
# only ever inherit the one(s) it actually needs --- never the whole keyring.
# These are the names ``scrubbed_env``'s ``keep`` argument selects from, and the
# values :func:`redact_secrets` masks out of anything that gets persisted.
MANAGED_VENDOR_KEYS = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "CODEX_API_KEY",
    "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "GROQ_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY",
)

# The ONLY non-secret variables a spawned CLI inherits. This is an ALLOWLIST on
# purpose: a blacklist can only drop the secrets we happened to think of, so a
# key the harness never heard of (a corporate SSO token, another tool's
# credential, a CI secret) would still be handed to a third-party binary. Names
# are matched case-insensitively because Windows upper-cases every environment
# key while POSIX shells conventionally use lowercase for the proxy vars.
_ENV_ALLOWLIST = frozenset({
    # process / OS basics --- Windows needs SystemRoot + ComSpec to start a
    # .cmd shim at all, and node reads the PROGRAM* / PROCESSOR_* vars.
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "OS",
    "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA",
    # user identity + where a CLI keeps its OWN config / cached login
    "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "APPDATA", "LOCALAPPDATA", "USER", "USERNAME", "LOGNAME", "SHELL",
    # scratch space
    "TEMP", "TMP", "TMPDIR",
    # locale / terminal --- these drive the CHILD's own output encoding
    "LANG", "LANGUAGE", "TZ", "TERM", "COLORTERM", "NO_COLOR",
    # network egress + corporate TLS trust stores
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    # the two supported CLIs' own config-LOCATION variables (paths, not secrets):
    # an operator who relocates codex's/opencode's config or cached login must
    # keep working now that unknown variables are withheld by default.
    "CODEX_HOME", "OPENCODE_CONFIG",
})

# Whole families that are location/locale-only and carry no credential.
_ENV_ALLOWED_PREFIXES = ("LC_", "XDG_")


def scrubbed_env(keep: tuple[str, ...] = ()) -> dict:
    """Minimal environment for a third-party CLI: an allowlist plus ``keep``.

    Only the non-secret operational variables in :data:`_ENV_ALLOWLIST` (plus the
    ``LC_*``/``XDG_*`` families) are forwarded. ``keep`` keeps its meaning: it
    names the credential(s) the CLI legitimately needs for the backend it is
    routing to, and those are re-admitted on top of the allowlist.

    Allowlist rather than blacklist: dropping only the names we know about still
    hands a third-party binary every OTHER secret in the operator's shell. Here
    an unknown variable is withheld by default, so the blast radius of spawning
    ``codex``/``opencode`` is one named credential instead of the whole
    environment.
    """
    keepset = {k.strip().upper() for k in keep if k and k.strip()}
    out: dict[str, str] = {}
    for name, val in os.environ.items():
        upper = name.upper()
        if (upper in _ENV_ALLOWLIST
                or upper.startswith(_ENV_ALLOWED_PREFIXES)
                or upper in keepset):
            out[name] = val
    return out


# --- secret redaction (anything below may be PERSISTED into llm_trace.jsonl) ---
# A failing CLI's stderr is quoted into CLIProviderError, RecordingProvider writes
# that message into the trace, and the trace is copied into the durable journal
# (memory/) --- which the README tells the user to COMMIT. So a stack trace or a
# debug dump that echoes a token would publish it. Mask credential shapes before
# the message is built rather than after it is on disk.
_SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{12,}"),        # OpenAI/DeepSeek/Anthropic
    re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{12,}"),      # GitHub
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{12,}"),                    # Slack
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),                        # AWS access key id
    re.compile(r"\bey[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
               r"\.[A-Za-z0-9_\-]+"),                                 # JWT
    # "Authorization: Bearer <token>" --- must run BEFORE the generic label rule
    # below, which would otherwise consume the word "Bearer" as the value and
    # leave the real token standing.
    re.compile(r"(?i)\b(bearer)(\s+)([A-Za-z0-9._\-]{8,})"),
    # generic "<label>=<value>" / '"label": "value"' dumps. An explicit ':'/'='
    # is REQUIRED: matching "<label> <word>" too would eat the diagnostic word
    # after it ("token expired" -> "token <redacted>"), and preflight keys its
    # auth-error detection off exactly those words. "bearer" is excluded as a
    # value so the pair above stays readable as "Authorization: Bearer <redacted>".
    re.compile(r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|"
               r"secret|password|passwd|authorization)"
               r"(\"?\s*[:=]\s*\"?)((?!bearer\b)[^\s\"',]+)"),
)

_REDACTED = "<redacted>"


def redact_secrets(text: str) -> str:
    """Mask credential shapes (and the operator's home path) in ``text``.

    Two passes, cheapest and most certain first:

      1. any *live* value of a :data:`MANAGED_VENDOR_KEYS` variable that appears
         verbatim --- an exact match, so it cannot be missed by pattern drift;
      2. known credential SHAPES (``sk-...``, ``ghp_...``, JWTs, ``token=...``),
         for secrets the harness does not manage.

    Finally the user's home directory is collapsed to ``~`` so a persisted error
    does not embed an absolute local path (the same leak that put machine paths
    into the committed CTX logs). Purely textual --- it never inspects a file.
    """
    if not text:
        return text
    secret_name = re.compile(
        r"(?i)(?:api[_-]?key|token|secret|password|passwd|credential|private[_-]?key)"
    )
    # Exact-value masking covers both managed provider credentials and unknown
    # secrets inherited by the harness itself (CI tokens, internal credentials,
    # etc.). The latter are intentionally not forwarded to child CLIs, but an
    # exception or model response can still echo one into a persisted artifact.
    for key, val in os.environ.items():
        if (key in MANAGED_VENDOR_KEYS or secret_name.search(key)) \
                and val and len(val) >= 8:
            text = text.replace(val, f"<{key} redacted>")
    for pat in _SECRET_PATTERNS:
        # A 3-group pattern is "<label><separator><value>": keep groups 1+2 so the
        # reader still sees WHAT was masked ("api_key=<redacted>"). A 0-group
        # pattern matches the bare secret, so the whole match goes.
        text = pat.sub(
            lambda m: (f"{m.group(1)}{m.group(2)}{_REDACTED}"
                       if m.re.groups >= 3 else _REDACTED),
            text,
        )
    try:
        home = str(pathlib.Path.home())
    except (RuntimeError, OSError):          # pragma: no cover - no home dir
        return text
    # Guard against a degenerate home ("/" for root in a container, "C:\" for a
    # misconfigured profile): substituting that would rewrite every path in the
    # message into nonsense, which is worse than the leak it prevents.
    if len(home) >= 4:
        for spelling in {home, home.replace("\\", "/")}:
            text = re.sub(re.escape(spelling), "~", text, flags=re.IGNORECASE)
    return text


class CLIProviderError(RuntimeError):
    """Raised when a CLI provider's binary is missing or the call fails."""


def ensure_binary(name: str) -> str:
    """Resolve ``name`` on PATH and return the ABSOLUTE path to spawn.

    Returning the resolved path (rather than just asserting it exists) is
    load-bearing on Windows: ``shutil.which`` honours ``PATHEXT`` and so finds an
    npm shim such as ``codex.CMD``, but ``CreateProcess`` --- what ``subprocess``
    ultimately calls --- only ever appends ``.exe``. Spawning the BARE name then
    dies with ``FileNotFoundError [WinError 2]`` even though the readiness check
    ("codex on PATH: ok") passed moments earlier. Spawning exactly what ``which``
    resolved keeps the doctor's verdict and the first real call consistent.
    """
    path = shutil.which(name)
    if path is None:
        raise CLIProviderError(
            f"'{name}' CLI not found on PATH. Install it (or use provider='scripted')."
        )
    return path


def join_prompt(system: str, messages: list[dict]) -> str:
    """CLIs take a single prompt --- fold system + user turns into one string."""
    parts = [system.strip()] if system.strip() else []
    parts += [m.get("content", "") for m in messages]
    return "\n\n".join(p for p in parts if p)


def run_cli(cmd: list[str], timeout: int = 600, env: dict | None = None,
            stdin: str | None = None) -> str:
    """Run a CLI provider's binary and return its stdout, decoded as UTF-8.

    The explicit ``encoding``/``errors`` are load-bearing, not tidiness. Plain
    ``text=True`` decodes with ``locale.getencoding()`` --- cp1252 on a stock
    Windows box --- and this is the one call in the package that captures the
    MODEL'S ANSWER. Model answers routinely contain the >= sign, an em dash,
    sigma, alpha: under cp1252 each UTF-8 byte is decoded as a separate Latin-1
    letter, so the answer comes back as mojibake, is persisted into
    ``llm_trace.jsonl`` and the journal, and poisons every later prompt --- and
    some byte sequences raise ``UnicodeDecodeError`` mid-read and kill the whole
    iteration. Decoding UTF-8 makes the recorded trace byte-identical across
    operating systems, which is what the bit-for-bit replay claim actually needs;
    ``errors='replace'`` turns a genuinely malformed byte into a visible U+FFFD
    instead of an exception.

    ``stdin`` is never inherited. When it is ``None`` the child gets
    ``DEVNULL``: an older opencode (<=1.17) meeting an ``ask`` permission in a
    non-interactive run BLOCKED waiting for a keypress on its inherited stdin,
    which a harness can never provide --- the run hung until the 600s timeout
    killed it. Detaching stdin makes that hang structurally impossible; current
    opencode (>=1.18) auto-rejects ``ask`` in non-interactive runs instead.
    """
    # cmd[0] is an absolute path (see ensure_binary), so it can carry the
    # operator's home directory into a persisted error message --- redact it.
    label = redact_secrets(str(cmd[0])) if cmd else "<empty command>"
    try:
        check_cancelled()
        proc = _cancellable_cli(cmd, timeout, env, stdin)
    except FileNotFoundError as exc:
        raise CLIProviderError(
            f"cannot run {label!r}: {redact_secrets(str(exc))}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CLIProviderError(f"{label} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        # Redact BEFORE truncating: cutting at 500 chars first could leave the
        # leading half of a token in the message the trace persists.
        err = redact_secrets((proc.stderr or "").strip())[:500]
        raise CLIProviderError(f"{label} exited {proc.returncode}: {err}")
    return proc.stdout


def start_owned_process(cmd, **kwargs):
    """Assign containment before a provider can create any descendants."""
    job = None
    proc = None
    if os.name == "nt":
        from .windows_job import WindowsJob
        job = WindowsJob()
    try:
        proc = subprocess.Popen(cmd, start_new_session=os.name != "nt",
                                creationflags=0x08000004 if os.name == "nt" else 0,
                                **kwargs)
        if job is not None:
            job.start(proc)
        return proc, job
    except BaseException:
        if job is not None:
            job.close()
        if proc is not None:
            proc.kill()
            proc.wait(timeout=10)
        raise


def _stop_tree(proc: subprocess.Popen, job=None) -> None:
    """Stop ONLY this provider's process tree (including CLI MCP children)."""
    if job is not None:
        job.close()
    else:  # pragma: no cover - Unix CI
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        proc.kill()
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        # Never wait indefinitely on pipes held outside the owned group.
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def _cancellable_cli(cmd, timeout, env, stdin):
    job = None
    proc = None
    deadline = time.monotonic() + timeout
    first = True
    try:
        proc, job = start_owned_process(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        check_cancelled()
        while True:
            check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout)
            try:
                stdout, stderr = proc.communicate(
                    input=stdin if first else None, timeout=min(0.1, remaining))
                check_cancelled()
                return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                first = False
    finally:
        if proc is not None:
            _stop_tree(proc, job)
