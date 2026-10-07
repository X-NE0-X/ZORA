"""First-run readiness checks + setup guidance (harness/preflight.py).

These tests never touch the network, a real SDK, or the vendored engine: every
probe (SDK import, CLI binary, engine import, prompt library) is a module-level
function, so they are stubbed here, and API-key presence is driven off
``os.environ`` with ``load_env=False`` so the vendored ``.env`` is never
consulted (the checks stay pinned to what the test controls).

Stubbing the engine probe is not just for speed: the real one imports the whole
vbt/numba/talib stack, so an unstubbed unit test would assert on whatever
happens to be installed on the machine running it.
"""
from __future__ import annotations

import contextlib
import io
import os

from harness import preflight as pf
from harness.config import RunConfig


@contextlib.contextmanager
def _env(key: str, value: str | None):
    """Temporarily set (value) or unset (None) an env var, restoring after."""
    had = key in os.environ
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if had:
            os.environ[key] = old            # type: ignore[assignment]
        else:
            os.environ.pop(key, None)


@contextlib.contextmanager
def _patch(name: str, fn):
    """Temporarily replace a preflight module attribute, restoring after."""
    old = getattr(pf, name)
    setattr(pf, name, fn)
    try:
        yield
    finally:
        setattr(pf, name, old)


@contextlib.contextmanager
def _runtime(*, packages: bool = True, engine_error: str = "",
             prompt_problems: tuple[str, ...] = ()):
    """Pin the LOCAL runtime probes (engine + prompts) to a known state."""
    with _patch("_engine_packages_present", lambda: packages), \
            _patch("_probe_engine", lambda: engine_error), \
            _patch("_probe_prompts", lambda: list(prompt_problems)):
        yield


# --- scripted -------------------------------------------------------------------
def test_scripted_always_ready():
    r = pf.check_provider("scripted")
    assert r.ok is True
    assert r.reason == ""
    assert r.steps == []


def test_check_config_default_is_scripted_and_ready():
    with _runtime():
        r = pf.check_config(RunConfig())
    assert r.provider == "scripted"
    assert r.ok is True
    # check_config is the RUN gate, so it covers the runtime too, not just the
    # credential (runtime-06)
    labels = [lbl for lbl, _ in r.checks]
    assert "vendored engine packages" in labels
    assert "prompt assets" in labels


# --- key providers: missing pieces ---------------------------------------------
def test_key_provider_not_ready_without_key():
    with _env("ANTHROPIC_API_KEY", None), _patch("_sdk_available", lambda m: True):
        r = pf.check_provider("claude", load_env=False)
    assert r.ok is False
    assert "ANTHROPIC_API_KEY" in r.reason
    # the guide must name the env var and point at the vendored .env file
    joined = "\n".join(r.steps)
    assert "ANTHROPIC_API_KEY" in joined
    assert pf.ENV_FILE_HINT in joined
    assert ("ANTHROPIC_API_KEY", False) in r.checks


def test_key_provider_not_ready_without_sdk():
    with _env("OPENAI_API_KEY", "sk-dummy"), _patch("_sdk_available", lambda m: False):
        r = pf.check_provider("openai", load_env=False)
    assert r.ok is False
    assert "openai" in r.reason and "SDK" in r.reason
    assert any("pip install openai" in s for s in r.steps)


def test_key_provider_ready_with_sdk_and_key():
    with _env("DEEPSEEK_API_KEY", "sk-dummy"), _patch("_sdk_available", lambda m: True):
        r = pf.check_provider("deepseek", load_env=False)
    assert r.ok is True
    assert ("DEEPSEEK_API_KEY", True) in r.checks


def test_deepseek_uses_openai_sdk():
    seen = []
    with _env("DEEPSEEK_API_KEY", "sk-dummy"), \
            _patch("_sdk_available", lambda m: seen.append(m) or True):
        pf.check_provider("deepseek", load_env=False)
    assert seen == ["openai"]                # DeepSeek speaks the OpenAI SDK


# --- secret hygiene -------------------------------------------------------------
def test_format_never_leaks_secret_value():
    secret = "sk-DO-NOT-LEAK-abcdef123456"
    with _env("ANTHROPIC_API_KEY", secret), _patch("_sdk_available", lambda m: True):
        r = pf.check_provider("claude", load_env=False)
        out = pf.format_readiness(r)
        out_markup = pf.format_readiness(r, markup=True)
    assert r.ok is True
    assert secret not in out
    assert secret not in out_markup
    assert secret not in r.reason
    assert secret not in "\n".join(r.steps + r.notes)


# --- CLI providers --------------------------------------------------------------
def test_cli_provider_not_ready_without_binary():
    with _patch("_binary_available", lambda n: False):
        r = pf.check_provider("codex", load_env=False)
    assert r.ok is False
    assert "codex" in r.reason
    assert any("codex login" in s for s in r.steps)
    assert ("codex on PATH", False) in r.checks


def test_cli_provider_ready_with_binary_but_warns_login():
    with _patch("_binary_available", lambda n: True):
        r = pf.check_provider("opencode", load_env=False)
    assert r.ok is True
    # login can't be verified cheaply -> surfaced as a reminder note, not a block
    assert any("opencode auth login" in n for n in r.notes)


# --- unknown provider -----------------------------------------------------------
def test_unknown_provider_not_ready():
    r = pf.check_provider("gemini", load_env=False)
    assert r.ok is False
    assert "unknown provider" in r.reason


# --- local runtime: the backtest engine + the prompt assets (runtime-06) ---------
def test_runtime_ready_when_engine_imports_and_prompts_load():
    with _runtime():
        r = pf.check_runtime()
    assert r.ok is True
    assert r.reason == ""
    assert ("vendored engine packages", True) in r.checks
    assert ("engine runtime dependencies", True) in r.checks
    assert ("prompt assets", True) in r.checks


def test_runtime_blocks_when_an_engine_dependency_is_missing():
    # the exact failure runtime-06 is about: doctor used to say "ready" and the
    # run died ~60s later on an InfraUnavailable traceback about talib/numba/...
    err = ("vendored infra is present but a runtime dependency is missing "
           "(No module named 'talib'); install the infra extras")
    with _runtime(engine_error=err):
        r = pf.check_runtime()
    assert r.ok is False
    assert err in r.reason                       # infra_engine's own wording
    assert ("engine runtime dependencies", False) in r.checks
    assert any("pip install" in s and "infra" in s for s in r.steps)


def test_runtime_blocks_when_the_engine_packages_are_absent():
    with _runtime(packages=False):
        r = pf.check_runtime()
    assert r.ok is False
    assert ("vendored engine packages", False) in r.checks
    assert "_vendor" in "\n".join(r.steps)
    # with the packages gone the import probe would only restate the same fact
    assert not any(lbl == "engine runtime dependencies" for lbl, _ in r.checks)


def test_runtime_blocks_when_a_prompt_asset_is_broken():
    with _runtime(prompt_problems=("prompt asset 'user_initial' not found at ...",)):
        r = pf.check_runtime()
    assert r.ok is False
    assert ("prompt assets", False) in r.checks
    assert any("user_initial" in s for s in r.steps)


def test_probe_engine_reuses_the_engines_own_diagnostic():
    # the probe must NOT re-list talib/numba/duckdb/vectorbt itself: it delegates
    # to infra_engine._ensure so the gate and the run can never disagree about
    # what is wrong.
    from harness import infra_engine

    def _boom() -> None:
        raise infra_engine.InfraUnavailable("engine says: duckdb is missing")

    old = infra_engine._ensure
    infra_engine._ensure = _boom
    try:
        assert pf._probe_engine() == "engine says: duckdb is missing"
    finally:
        infra_engine._ensure = old


def test_probe_engine_survives_a_hard_import_failure():
    # a half-installed native dependency can raise almost anything on import; a
    # probe reports the reason, it never crashes the doctor.
    from harness import infra_engine

    def _boom() -> None:
        raise OSError("DLL load failed while importing _ta_lib")

    old = infra_engine._ensure
    infra_engine._ensure = _boom
    try:
        assert "DLL load failed" in pf._probe_engine()
    finally:
        infra_engine._ensure = old


def test_declared_numpy_and_pandas_bounds_block_doctor():
    original = pf.importlib.metadata.version
    versions = {"numpy": "2.3.0", "pandas": "2.2.3"}
    pf.importlib.metadata.version = lambda name: versions[name]
    try:
        err = pf._probe_declared_versions()
    finally:
        pf.importlib.metadata.version = original
    assert "numpy 2.3.0 does not satisfy >=2.4.6,<2.5" in err
    assert "pandas 2.2.3 does not satisfy >=3.0.3,<4.0" in err


# --- the combined gate -----------------------------------------------------------
def test_check_all_fails_when_only_the_runtime_is_broken():
    # scripted needs no credential, but it still needs the engine -> the gate
    # must block it exactly as a missing API key would.
    with _runtime(engine_error="numba is missing"):
        r = pf.check_all("scripted")
    assert r.ok is False
    assert "numba is missing" in r.reason
    assert r.scope == "run"
    # and the headline must not blame the provider for the engine's problem
    assert "local runtime" in pf.format_readiness(r).splitlines()[0]


def test_check_all_fails_when_only_the_credential_is_missing():
    with _runtime(), _env("ANTHROPIC_API_KEY", None), \
            _patch("_sdk_available", lambda m: True):
        r = pf.check_all("claude", load_env=False)
    assert r.ok is False
    assert "ANTHROPIC_API_KEY" in r.reason


def test_check_all_ready_needs_both_halves():
    with _runtime(), _env("OPENAI_API_KEY", "sk-dummy"), \
            _patch("_sdk_available", lambda m: True):
        r = pf.check_all("openai", load_env=False)
    assert r.ok is True
    assert [lbl for lbl, _ in r.checks][:2] == ["openai SDK", "OPENAI_API_KEY"]


def test_explain_failure_stays_provider_only():
    # a lapsed login is not a runtime problem: the guide must keep talking about
    # the credential, and must not drag the engine probe into an error path.
    with _runtime(engine_error="numba is missing"), \
            _patch("_binary_available", lambda n: True):
        guide = pf.explain_failure("codex", RuntimeError("401 unauthorized"))
    assert guide is not None
    assert "numba" not in guide.reason
    assert any("codex login" in s for s in guide.steps)


# --- format_all -----------------------------------------------------------------
def test_format_all_lists_every_provider():
    with _runtime(), _patch("_sdk_available", lambda m: False), \
            _patch("_binary_available", lambda n: False):
        out = pf.format_all(load_env=False)
    for p in pf.ALL_PROVIDERS:
        assert p in out
    assert "scripted" in out


def test_format_all_reports_the_runtime_once():
    with _runtime(engine_error="numba is missing"), \
            _patch("_sdk_available", lambda m: True), \
            _patch("_binary_available", lambda n: True):
        out = pf.format_all(load_env=False)
    # the engine is the same for every provider -> stated once, not six times
    assert out.count("numba is missing") == 1
    assert "local runtime" in out


# --- the CLI surface (doctor / run) ----------------------------------------------
def test_doctor_all_exits_nonzero_when_the_runtime_is_broken():
    # runtime-06: 'doctor' used to print "ready" and exit 0 on a machine where
    # no run could possibly work.
    from harness import cli

    buf = io.StringIO()
    with _runtime(engine_error="No module named 'vectorbt'"), \
            contextlib.redirect_stdout(buf):
        code = cli.main(["doctor", "--all"])
    assert code == 1
    assert "vectorbt" in buf.getvalue()


def test_doctor_provider_reports_the_runtime_too():
    from harness import cli

    buf = io.StringIO()
    with _runtime(engine_error="No module named 'talib'"), \
            _patch("_binary_available", lambda n: True), \
            contextlib.redirect_stdout(buf):
        code = cli.main(["doctor", "--provider", "codex"])
    assert code == 1
    assert "talib" in buf.getvalue()


def test_run_is_blocked_by_a_broken_runtime_even_for_scripted():
    # the scripted provider needs no credential, so this path used to skip the
    # gate entirely and fail deep inside the backtest.
    from harness import cli

    buf = io.StringIO()
    with _runtime(engine_error="No module named 'talib'"), \
            contextlib.redirect_stdout(buf):
        code = cli.main(["run", "--run-name", "runtime_gate_test"])
    assert code == 3
    assert "talib" in buf.getvalue()


def test_format_readiness_markup_has_no_bare_brackets_from_checks():
    # RichLog(markup=True) parses [..] tags; our check markers must not emit
    # literal square brackets that Rich would try (and fail) to parse.
    with _patch("_binary_available", lambda n: False):
        r = pf.check_provider("codex", load_env=False)
    out = pf.format_readiness(r, markup=True)
    # only well-formed Rich tags remain; no "[x]" / "[ ]" style checkboxes
    assert "[x]" not in out and "[ ]" not in out


# --- explain_failure ------------------------------------------------------------
def test_explain_failure_ignores_non_auth_errors():
    exc = ValueError("no valid active factor proposal was produced")
    assert pf.explain_failure("openai", exc) is None


def test_explain_failure_ignores_scripted():
    exc = RuntimeError("401 unauthorized")
    assert pf.explain_failure("scripted", exc) is None


def test_explain_failure_key_provider_missing_key():
    with _env("ANTHROPIC_API_KEY", None), _patch("_sdk_available", lambda m: True):
        guide = pf.explain_failure(
            "claude", RuntimeError("ANTHROPIC_API_KEY is not set"))
    assert guide is not None and guide.ok is False
    assert "ANTHROPIC_API_KEY" in "\n".join(guide.steps + [guide.reason])


def test_explain_failure_cli_login_when_binary_present():
    # binary present so preflight passes, but the run hit a 401 -> login guide
    with _patch("_binary_available", lambda n: True):
        guide = pf.explain_failure("codex", RuntimeError("codex exited 1: not signed in"))
    assert guide is not None and guide.ok is False
    assert any("codex login" in s for s in guide.steps)


def test_looks_like_auth_error():
    assert pf.looks_like_auth_error("Error 401: Unauthorized")
    assert pf.looks_like_auth_error("ANTHROPIC_API_KEY is not set")
    assert pf.looks_like_auth_error("please sign in first")
    assert not pf.looks_like_auth_error("connection reset by peer")
    assert not pf.looks_like_auth_error("uncomputable formula")
