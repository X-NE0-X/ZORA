"""First-run readiness checks + setup guidance --- credentials AND local runtime.

The ``scripted`` provider and the ``synthetic`` data source run with no key and
no network, so the harness works out of the box. The moment a *real* provider is
selected (``claude``, ``openai``, ``deepseek``, ``codex``, ``opencode``) it needs
a credential a first-time user has almost certainly not set up yet --- an API key,
or a signed-in CLI. Without this module a run dies deep inside the provider with a
bare ``RuntimeError('ANTHROPIC_API_KEY is not set')`` or a CLI's raw non-zero exit.

A credential is only half of "can this run at all", though. EVERY run --- including
a fully offline scripted/synthetic one --- goes through the vendored vbt backtest
engine and renders its prompts from ``harness/prompts``. A missing engine
dependency (talib / numba / duckdb / vectorbt) or a hand-broken prompt file kills
the run about a minute in, which is exactly the class of failure this module
exists to pre-empt. So the gate has two halves:

  * :func:`check_provider` --- the credential (SDK importable, key set, CLI on PATH)
  * :func:`check_runtime`  --- the local runtime (engine importable, prompts loadable)

and :func:`check_all` / :func:`check_config` combine them into the single
:class:`Readiness` that both the CLI and the TUI gate a run on. Every check is a
*cheap, reliable* signal taken BEFORE a run starts, and a failure carries an
actionable, step-by-step setup guide.

A CLI's *login state* cannot be probed cheaply or honestly (it would mean running
the CLI, whose status command varies by version), so it is surfaced as a reminder
note; and if a run later fails with an auth-shaped error, :func:`explain_failure`
turns that failure into the same login guide.

Security: nothing here ever reads or prints a secret VALUE --- only whether a key
is *set* (``bool(os.environ.get(...))``). The formatted guide names the env var,
never its contents.
"""
from __future__ import annotations

import importlib.util
import importlib.metadata
import os
import platform
import shutil
from dataclasses import dataclass, field

from . import palette as _pal
from . import promptlib
from . import env as _env
from .env import load_provider_env


def _env_file_hint() -> str:
    """Display path for the .env this installation actually reads and writes.

    A source checkout / ``pip install -e .`` keeps the repo-relative vendored
    file, and naming it relatively reads better in the setup guide. A plain wheel
    install writes to a per-user config dir instead (site-packages is wiped by
    ``pip install -U`` and unwritable on a system Python), so there the guide must
    print the real absolute path or it would send the user to a file that does
    not exist. Hardcoding the vendored string was correct only for the first case.
    """
    if _env.is_source_checkout():
        return "harness/_vendor/ENV_MGMT/.env"
    return str(_env.resolve_env_path())


ENV_FILE_HINT = _env_file_hint()

# API-key providers: need their SDK importable AND their key present in the env.
KEY_PROVIDERS: dict[str, dict[str, str]] = {
    "claude": {
        "label": "Anthropic (Claude)",
        "sdk": "anthropic",
        "sdk_install": "pip install anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "console": "https://console.anthropic.com/  (Settings -> API Keys)",
    },
    "openai": {
        "label": "OpenAI",
        "sdk": "openai",
        "sdk_install": "pip install openai",
        "env_key": "OPENAI_API_KEY",
        "console": "https://platform.openai.com/api-keys",
    },
    "deepseek": {
        "label": "DeepSeek",
        "sdk": "openai",                     # DeepSeek speaks the OpenAI SDK
        "sdk_install": "pip install openai",
        "env_key": "DEEPSEEK_API_KEY",
        "console": "https://platform.deepseek.com/  (API Keys)",
    },
}

# CLI providers: need their binary on PATH AND a one-time login (not key-based).
CLI_PROVIDERS: dict[str, dict[str, str]] = {
    "codex": {
        "label": "OpenAI Codex CLI",
        "binary": "codex",
        "login_cmd": "codex login",
        "login_desc": "ChatGPT sign-in -- no API key needed",
    },
    "opencode": {
        "label": "opencode CLI",
        "binary": "opencode",
        "login_cmd": "opencode auth login",
        "login_desc": "the opencode CLI's own auth",
    },
}

ALL_PROVIDERS: tuple[str, ...] = (
    ("scripted",) + tuple(KEY_PROVIDERS) + tuple(CLI_PROVIDERS)
)


@dataclass
class Readiness:
    """The outcome of a readiness check + how to fix it if not ready."""

    provider: str
    ok: bool
    reason: str = ""                                  # one-line "what's missing"
    steps: list[str] = field(default_factory=list)    # ordered setup instructions
    notes: list[str] = field(default_factory=list)    # non-blocking reminders
    checks: list[tuple[str, bool]] = field(default_factory=list)  # (label, ok) rows
    # What was checked, so the headline can't blame the provider for a broken
    # engine: "provider" (credential only), "runtime" (engine + prompts) or
    # "run" (both, the gate a run is actually held to).
    scope: str = "provider"


def _sdk_available(module: str) -> bool:
    """True if ``module`` can be imported, without importing it (cheap, no side effects).

    Split out as a module-level function so tests can stub it (the real SDKs are
    optional deps that may be absent in a bare offline environment).
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):        # ValueError: bad/partial package state
        return False


def _binary_available(name: str) -> bool:
    return shutil.which(name) is not None


def _doctor_hint(provider: str) -> str:
    return f"re-check:          python -m harness.cli doctor --provider {provider}"


def _check_key_provider(provider: str) -> Readiness:
    spec = KEY_PROVIDERS[provider]
    sdk_ok = _sdk_available(spec["sdk"])
    key_ok = bool(os.environ.get(spec["env_key"]))
    checks = [(f"{spec['sdk']} SDK", sdk_ok), (spec["env_key"], key_ok)]

    if sdk_ok and key_ok:
        return Readiness(
            provider, ok=True, checks=checks,
            notes=[f"{spec['label']} ready: {spec['env_key']} is set "
                   "(its value is never shown)."],
        )

    reasons: list[str] = []
    steps: list[str] = []
    if not sdk_ok:
        reasons.append(f"the '{spec['sdk']}' SDK is not installed")
        steps.append(f"install the SDK:   {spec['sdk_install']}")
    if not key_ok:
        reasons.append(f"{spec['env_key']} is not set")
        steps.append(f"get a key:         {spec['console']}")
        steps.append("then set it either")
        steps.append(f"   shell (Git Bash):  export {spec['env_key']}=<your-key>")
        steps.append(f"   or file:           add  {spec['env_key']}=<your-key>  "
                     f"to  {ENV_FILE_HINT}")
        steps.append(f"                      (copy {ENV_FILE_HINT}.template; "
                     "a real shell var always wins)")
    steps.append(_doctor_hint(provider))
    return Readiness(provider, ok=False, reason="; ".join(reasons),
                     steps=steps, checks=checks)


def _check_cli_provider(provider: str) -> Readiness:
    spec = CLI_PROVIDERS[provider]
    bin_ok = _binary_available(spec["binary"])
    checks = [(f"{spec['binary']} on PATH", bin_ok)]

    if not bin_ok:
        return Readiness(
            provider, ok=False,
            reason=f"'{spec['binary']}' CLI not found on PATH",
            steps=[
                f"install the {spec['label']} and put '{spec['binary']}' on your PATH",
                f"sign in:           {spec['login_cmd']}   ({spec['login_desc']})",
                _doctor_hint(provider),
            ],
            checks=checks,
        )
    # Binary present, but login state can't be verified cheaply/honestly. Pass the
    # gate and surface the one-time sign-in as a reminder; explain_failure() turns
    # a later auth-shaped run failure into the concrete login guide.
    return Readiness(
        provider, ok=True, checks=checks,
        notes=[
            f"'{spec['binary']}' found on PATH.",
            f"first time? sign in once:  {spec['login_cmd']}   ({spec['login_desc']}).",
            "if a run fails with an auth error, that sign-in is the fix.",
        ],
    )


def check_provider(provider: str, model: str | None = None, *,
                   load_env: bool = True) -> Readiness:
    """Readiness of a single provider by name (``model`` is accepted for symmetry).

    ``load_env=True`` first loads the vendored ``.env`` (as a real run would) so a
    key stored there counts; tests pass ``load_env=False`` to keep the check pinned
    to the process environment they control.
    """
    if provider == "scripted":
        return Readiness(
            provider, ok=True,
            checks=[("offline (no credential needed)", True)],
            notes=["scripted provider: offline canned responses, no key or login."],
        )
    if provider in KEY_PROVIDERS:
        if load_env:
            load_provider_env()
        return _check_key_provider(provider)
    if provider in CLI_PROVIDERS:
        return _check_cli_provider(provider)
    return Readiness(
        provider, ok=False,
        reason=f"unknown provider {provider!r}",
        steps=[f"choose one of: {', '.join(ALL_PROVIDERS)}"],
        checks=[("known provider", False)],
    )


# --- local runtime: the backtest engine + the prompt assets ----------------------
# Name used in the Readiness.provider slot when the runtime is checked on its own
# (it is not a provider, but the dataclass field is what the renderer prints).
RUNTIME_LABEL = "local runtime"


def _engine_packages_present() -> bool:
    """True if the vendored engine packages are on disk (no import attempted).

    Module-level so tests can stub it; imported lazily because :mod:`infra_engine`
    pulls in pandas/numpy and the TUI imports this module at startup.
    """
    from . import infra_engine

    return infra_engine.available()


def _probe_engine() -> str:
    """Import-probe the vendored engine; ``""`` if usable, else the reason.

    The probe is ``infra_engine._ensure`` itself --- the module that OWNS the
    dependency knowledge --- so the user reads the same actionable
    ``InfraUnavailable`` text the run would have raised, and the two can never
    disagree. Re-listing talib/numba/duckdb/vectorbt here would drift the moment
    the engine grows a dependency.

    It costs a few seconds (it really imports the stack), which is the point: the
    alternative is discovering the missing dependency a minute into a run. For a
    real run the cost is not even extra --- the backtest imports the same stack.
    Any exception is a failed probe (a half-installed native dependency can raise
    almost anything on import), never a crashed doctor.
    """
    from . import infra_engine

    try:
        infra_engine._ensure()
    except Exception as exc:            # noqa: BLE001 - a probe reports, never raises
        return str(exc)
    return _probe_declared_versions()


def _probe_declared_versions() -> str:
    """Return any Python/numpy/pandas drift from ``pyproject.toml`` bounds."""
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError as exc:
        return (f"cannot verify declared runtime versions ({exc}); install the "
                "project dependencies with: pip install .")

    declared = (
        ("Python", platform.python_version(), ">=3.11,<3.15"),
        ("numpy", importlib.metadata.version("numpy"), ">=2.4.6,<2.5"),
        ("pandas", importlib.metadata.version("pandas"), ">=3.0.3,<4.0"),
    )
    drift = [f"{name} {installed} does not satisfy {spec}"
             for name, installed, spec in declared
             if Version(installed) not in SpecifierSet(spec)]
    return "; ".join(drift)


def _probe_prompts() -> list[str]:
    """Problems with the prompt library (``[]`` = every required prompt loads)."""
    return promptlib.verify()


def check_runtime() -> Readiness:
    """Readiness of the LOCAL runtime: backtest engine + prompt assets.

    Blocks a run exactly like a missing credential does, because the outcome is
    the same: the run cannot produce a verdict. A missing engine dependency used
    to surface as an ``InfraUnavailable`` traceback ~60s in (after the model had
    already been paid for a proposal), and a renamed prompt file as a repeated
    "provider error" ending in a misleading "no valid active factor proposal".
    """
    checks: list[tuple[str, bool]] = []
    reasons: list[str] = []
    steps: list[str] = []
    notes: list[str] = []

    pkgs_ok = _engine_packages_present()
    checks.append(("vendored engine packages", pkgs_ok))
    if not pkgs_ok:
        reasons.append("the vendored engine packages "
                       "(CTX / FactorEngine / BacktestEngine) are missing")
        steps.append("re-install the harness with its _vendor tree intact "
                     "(harness/harness/_vendor/)")
    else:
        # Only meaningful once the packages exist: with them absent the import
        # probe would just restate the line above.
        engine_err = _probe_engine()
        checks.append(("engine runtime dependencies", not engine_err))
        if engine_err:
            reasons.append(engine_err)
            steps.append("install the engine's runtime deps:  "
                         "pip install '.[infra]'")
            steps.append("   (TA-Lib also needs its native C library on the system)")

    problems = _probe_prompts()
    checks.append(("prompt assets", not problems))
    if problems:
        reasons.append(f"{len(problems)} prompt asset(s) missing or malformed")
        steps.append("restore the prompt files under harness/prompts/ "
                     "(a hand-edit or a rename is the usual cause):")
        steps.extend(f"   {p}" for p in problems)

    ok = not reasons
    if ok:
        notes.append("backtest engine imports and all "
                     f"{len(promptlib.REQUIRED_PROMPTS)} prompt assets load.")
    return Readiness(RUNTIME_LABEL, ok=ok, reason="; ".join(reasons),
                     steps=steps, notes=notes, checks=checks, scope="runtime")


# --- the combined gate -----------------------------------------------------------
def _merge(*parts: Readiness) -> Readiness:
    """Fold several readiness results into the one outcome a run is gated on.

    ``ok`` only if EVERY part is ok; rows, steps and notes concatenate in order so
    the printed block reads credential-first, then runtime. The provider name of
    the first part is kept: the caller asked "can I run with this provider?", and
    the answer should still say which provider it is about.
    """
    return Readiness(
        provider=parts[0].provider,
        ok=all(p.ok for p in parts),
        reason="; ".join(p.reason for p in parts if p.reason and not p.ok),
        steps=[s for p in parts for s in p.steps],
        notes=[n for p in parts for n in p.notes],
        checks=[c for p in parts for c in p.checks],
        scope="run",
    )


def check_all(provider: str, model: str | None = None, *,
              load_env: bool = True) -> Readiness:
    """Everything a run needs: the provider credential AND the local runtime.

    This is the gate: ``cli.cmd_run`` (and the TUI's Run / Ctrl+D) must use this
    rather than :func:`check_provider`, so that an unrunnable backtest engine
    blocks a run the same way a missing API key does --- including for
    ``scripted``, which needs no credential but still needs the engine.
    """
    return _merge(check_provider(provider, model, load_env=load_env),
                  check_runtime())


def check_config(config, *, load_env: bool = True) -> Readiness:
    """Readiness for a RunConfig: its provider's credential + the local runtime."""
    from .providers.codex_controls import validate_config
    try:
        validate_config(config)
    except ValueError as exc:
        return Readiness(config.provider, ok=False, reason=str(exc),
                         checks=[("contract / model / reasoning validity", False)], scope="run")
    ready = check_all(getattr(config, "provider", "scripted"),
                      getattr(config, "model", None), load_env=load_env)
    if config.provider == "codex":
        ready.checks.append(("contract / model / reasoning validity", True))
        ready.notes.append("Codex uses explicit model selection; LLM seed, temperature and max_tokens are unsupported. Seed still controls local data/engine reproducibility.")
    return ready


# --- rendering ------------------------------------------------------------------
def _c(text: str, color: str, markup: bool) -> str:
    """Tag ``text`` with a Rich colour when rendering into the TUI's RichLog.

    Colours come from :mod:`harness.palette`, not from terminal colour names, so
    this block sits inside the TUI's design language instead of painting whatever
    "green" the terminal happens to have. ``markup=False`` (the CLI) is untouched.
    """
    return f"[{color}]{text}[/]" if markup else text


def format_readiness(r: Readiness, *, markup: bool = False) -> str:
    """Human-readable readiness block. ``markup=True`` emits Rich tags for the TUI.

    Kept free of literal square brackets so it is safe to write into a
    ``RichLog(markup=True)`` without accidental tag parsing.
    """
    what = {
        # the runtime is not a provider, so it is named plainly...
        "runtime": RUNTIME_LABEL,
        # ...and a combined check must not read as if the PROVIDER needed setup
        # when it is the engine or the prompt library that is broken.
        "run": f"provider '{r.provider}' + {RUNTIME_LABEL}",
    }.get(r.scope, f"provider '{r.provider}'")
    lines: list[str] = []
    if r.ok:
        lines.append(_c(f"OK  {what}: ready", _pal.OK, markup))
    else:
        lines.append(_c(f"!!  {what}: needs setup", _pal.WARN, markup))

    for label, ok in r.checks:
        mark = _c("ok", _pal.OK, markup) if ok else _c("XX", _pal.BAD, markup)
        suffix = "" if ok else _c("  (missing)", _pal.BAD, markup)
        lines.append(f"    {mark}  {label}{suffix}")

    if r.reason and not r.ok:
        lines.append("")
        lines.append(_c(f"  what's missing: {r.reason}", _pal.BAD, markup))

    if r.steps:
        lines.append("")
        lines.append(_c("  set it up:", "bold", markup))
        for s in r.steps:
            # a step that starts with whitespace is a continuation / sub-line
            lines.append(f"    {s}" if s[:1] == " " else f"  - {s}")

    for n in r.notes:
        lines.append(f"  {_c('note:', _pal.NOTE, markup)} {n}")

    return "\n".join(lines)


def format_all(*, markup: bool = False, load_env: bool = True) -> str:
    """One compact status line per provider + the runtime block (``doctor --all``).

    The runtime is printed once, not per provider: it is the same engine and the
    same prompt library whichever provider is selected, and a run is impossible
    without it regardless of how many credentials are in place.
    """
    lines = [_c("provider readiness (scripted needs nothing):", "bold", markup)]
    for p in ALL_PROVIDERS:
        r = check_provider(p, load_env=load_env)
        if r.ok:
            state = _c("ready      ", _pal.OK, markup)
            tail = ""
        else:
            state = _c("needs setup", _pal.WARN, markup)
            tail = f"  <- {r.reason}"
        lines.append(f"  {p:<9}  {state}{tail}")
    lines.append("")
    lines.append(format_readiness(check_runtime(), markup=markup))
    lines.append("")
    lines.append("details + fix steps:  python -m harness.cli doctor --provider <name>")
    return "\n".join(lines)


# --- runtime failure translation ------------------------------------------------
# Substrings that mark an error as a credential / login problem rather than a bug.
# Deliberately specific to avoid firing on unrelated failures.
_AUTH_HINTS = (
    "login", "log in", "sign in", "signin", "not signed in",
    "unauthor", "authentication", "not authenticated",
    "401", "403", "api key", "api_key", "apikey",
    "credential", "is not set", "invalid key", "expired",
)


def looks_like_auth_error(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in _AUTH_HINTS)


def explain_failure(provider: str, exc, model: str | None = None) -> Readiness | None:
    """Turn an auth-shaped run failure into a :class:`Readiness` guide, else ``None``.

    Used in the ``run``/TUI error path as a safety net for the case preflight can't
    verify up front: a CLI whose binary is present but whose login has lapsed (or a
    key that is present but invalid/expired). Non-auth errors return ``None`` so the
    caller re-raises them as the genuine bugs they are.
    """
    if provider == "scripted":
        return None
    if not looks_like_auth_error(str(exc)):
        return None

    r = check_provider(provider, model, load_env=False)
    if not r.ok:
        return r          # preflight already knows exactly what's missing

    # Preflight passed (e.g. CLI binary found, or key present) but the run still hit
    # an auth wall -> it's a login / bad-credential problem. Emit the targeted fix.
    if provider in CLI_PROVIDERS:
        spec = CLI_PROVIDERS[provider]
        return Readiness(
            provider, ok=False,
            reason=f"the run failed with an auth-shaped error -- you may not be "
                   f"signed in to '{spec['binary']}'",
            steps=[f"sign in:           {spec['login_cmd']}   ({spec['login_desc']})",
                   _doctor_hint(provider)],
            checks=r.checks,
        )
    if provider in KEY_PROVIDERS:
        spec = KEY_PROVIDERS[provider]
        return Readiness(
            provider, ok=False,
            reason=f"the run failed with an auth-shaped error -- {spec['env_key']} "
                   "may be invalid or expired",
            steps=[f"check / replace the key:  {spec['console']}",
                   f"set it in {ENV_FILE_HINT} or  export {spec['env_key']}=<your-key>",
                   _doctor_hint(provider)],
            checks=r.checks,
        )
    return None
