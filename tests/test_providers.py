"""Provider registry + graceful missing-dependency behaviour + interp parsing,
plus the two portability properties the CLI-backed providers depend on:

  * the model's answer is decoded as UTF-8, not as the machine's locale codepage
    (portability-01) --- otherwise the recorded trace differs across operating
    systems and the bit-for-bit replay claim is false;
  * a CLI is spawned by the path ``shutil.which`` resolved, not by its bare name
    (portability-02) --- on Windows ``which`` finds a ``.cmd`` npm shim that
    ``CreateProcess`` then cannot launch, so ``doctor`` would say "ready" and the
    first call would die with WinError 2.

...and the offline demo's contract: the default script is as long as the default
search depth, so a keyless first run shows eight distinct ideas instead of
silently reprinting one (gap-04).
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings
import pytest

from harness.config import RunConfig
from harness.providers import PROVIDERS, get_provider
from harness.providers.cli_base import CLIProviderError, ensure_binary, run_cli
from harness.providers.scripted import DEFAULT_SCRIPT, ScriptedProvider
from harness.proposer import ProposalError, parse_proposal


def test_registry_contents():
    assert set(PROVIDERS) == {
        "scripted", "claude", "openai", "deepseek", "codex", "opencode"
    }
    assert isinstance(get_provider("scripted"), ScriptedProvider)


def test_unknown_provider_raises():
    try:
        get_provider("does-not-exist")
    except ValueError:
        return
    raise AssertionError("unknown provider should raise ValueError")


def test_optional_providers_fail_cleanly_when_unavailable():
    # Missing SDK / key / binary must raise RuntimeError (or a subclass),
    # never a bare ImportError/FileNotFoundError leaking through.
    for name in ("claude", "openai", "deepseek", "codex", "opencode"):
        try:
            get_provider(name)
        except RuntimeError:
            pass                     # expected when the dep/key/binary is absent
        except Exception as exc:     # noqa: BLE001
            raise AssertionError(
                f"{name} raised {type(exc).__name__}, expected RuntimeError"
            ) from exc
        # if it constructed, the dep really is installed --- also fine


def test_scripted_drives_a_completion():
    p = get_provider("scripted")
    r = p.complete("system", [{"role": "user", "content": "hi"}], model="x")
    assert r.provider == "scripted"
    assert '"formula"' in r.text and r.prompt_hash


# --- gap-04: the offline demo must not silently repeat itself ---------------
def test_default_script_is_as_long_as_the_default_search_depth():
    """`python -m harness.cli run` with no key is the ONLY thing a new user can
    run, and it uses DEFAULT_SCRIPT with the default max_iters. A script shorter
    than the depth makes the provider hold on its last response, so the demo
    reprinted one formula for five straight rounds and looked hung."""
    assert len(DEFAULT_SCRIPT) == RunConfig().max_iters, (
        f"DEFAULT_SCRIPT has {len(DEFAULT_SCRIPT)} responses but max_iters "
        f"defaults to {RunConfig().max_iters}: the demo would repeat itself"
    )


def test_default_script_entries_are_distinct_and_parseable():
    formulas = []
    provider = ScriptedProvider()
    for _ in range(RunConfig().max_iters):
        response = provider.complete("system", [], model="scripted")
        p = parse_proposal(response.text, require_interpretability=True)
        formulas.append(p.formula)
    assert len(set(formulas)) == len(formulas), \
        f"the demo must show a trajectory, not repeats: {formulas}"


def test_scripted_flags_exhaustion_instead_of_repeating_silently():
    prov = ScriptedProvider(script=["a", "b"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = [prov.complete("s", [], model="m") for _ in range(2)]
        assert not caught, "no warning while the script still has responses"
        assert [r.meta["exhausted"] for r in first] == [False, False]

        held = [prov.complete("s", [], model="m") for _ in range(2)]
    assert [r.text for r in held] == ["b", "b"]          # still holds (not fatal)
    assert all(r.meta["exhausted"] for r in held)
    msgs = [str(w.message) for w in caught if w.category is RuntimeWarning]
    assert len(msgs) == 1, "warn once per provider, not once per call"
    assert "exhausted" in msgs[0] and "max_iters=2" in msgs[0]


def _proposal(rationale, mechanism):
    return json.dumps({
        "formula": "-delta(close,w)",
        "parameters": {"w": {"type": "Window", "value": 5}},
        "rationale": rationale,
        "mechanism": mechanism,
        "expected_sign": -1,
    })


def test_interpretability_parsing_enforced():
    good = _proposal(
        "Short-term reversal from retail overreaction to recent 5-day moves.",
        "Liquidity providers earn the spread as prices mean-revert to fair value.",
    )
    p = parse_proposal(good, require_interpretability=True)
    assert p.mechanism and p.rationale

    thin = _proposal("reversal", "")
    try:
        parse_proposal(thin, require_interpretability=True)
    except ProposalError:
        pass
    else:
        raise AssertionError("thin rationale/mechanism must be rejected when required")

    # when not required, a thin proposal is fine
    parse_proposal(thin, require_interpretability=False)




def test_interpretability_rejects_nondistinct_fields():
    # rationale == mechanism must be rejected when interpretability is required
    same = "Short-term reversal from retail overreaction to recent 5-day moves."
    dup = json.dumps({"formula": "-delta(close,w)", "parameters": {"w": {"type": "Window", "value": 5}}, "rationale": same,
                      "mechanism": same, "expected_sign": -1})
    try:
        parse_proposal(dup, require_interpretability=True)
    except ProposalError:
        pass
    else:
        raise AssertionError("identical rationale/mechanism must be rejected")

    # a field that just restates the formula is rejected too (use a long formula
    # so this trips the parrot check, not the min-length floor)
    long_formula = "-rank(ts_mean(returns,lookback)/ts_std(returns,lookback))"
    parroted = json.dumps({
        "formula": long_formula,
        "parameters": {"lookback": {"type": "Window", "value": 20}},
        "rationale": long_formula,
        "mechanism": "Liquidity providers earn the spread as prices mean-revert.",
        "expected_sign": -1,
    })
    try:
        parse_proposal(parroted, require_interpretability=True)
    except ProposalError:
        pass
    else:
        raise AssertionError("rationale restating the formula must be rejected")


# --- portability-01/02: the CLI provider path -------------------------------
# A fake CLI: a PATH-resolvable shim (a .cmd on Windows, exactly like the npm
# shims codex/opencode really install) that execs a python script. That gives a
# genuine end-to-end run through ensure_binary -> subprocess -> run_cli without
# needing either real CLI installed.
_NON_ASCII = "rank(x) ≥ 0 — sigma σ done"


def _install_fake_cli(directory, name, py_source):
    d = pathlib.Path(directory)
    script = d / f"{name}_impl.py"
    script.write_text(py_source, encoding="utf-8")
    if os.name == "nt":
        shim = d / f"{name}.cmd"
        shim.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
                        encoding="ascii")
    else:
        shim = d / name
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                        encoding="ascii")
        shim.chmod(0o755)
    return shim


def _with_path(directory):
    """Prepend ``directory`` to PATH; returns the previous value to restore."""
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = str(directory) + os.pathsep + old
    return old


def test_ensure_binary_returns_the_resolved_path_not_the_bare_name():
    with tempfile.TemporaryDirectory() as d:
        shim = _install_fake_cli(d, "zora_probe_cli", "print('probe-ok')\n")
        old = _with_path(d)
        try:
            resolved = ensure_binary("zora_probe_cli")
            assert os.path.isabs(resolved), resolved
            assert os.path.exists(resolved)
            assert pathlib.Path(resolved).samefile(shim)
            # the resolved path is spawnable...
            assert "probe-ok" in run_cli([resolved])
            if os.name == "nt":
                # ...and on Windows the BARE name is not: shutil.which honours
                # PATHEXT and finds the .cmd, but CreateProcess only appends
                # .exe. This is exactly the gap that let `doctor` report "ready"
                # and the first completion die with WinError 2.
                try:
                    subprocess.run(["zora_probe_cli"], capture_output=True)
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError(
                        "expected the bare .cmd name to be unspawnable on Windows"
                    )
        finally:
            os.environ["PATH"] = old


def test_ensure_binary_reports_a_missing_binary_cleanly():
    try:
        ensure_binary("zora_definitely_not_installed_cli")
    except CLIProviderError as exc:
        assert "not found on PATH" in str(exc)
    else:
        raise AssertionError("a missing binary must raise CLIProviderError")


def test_cli_providers_spawn_the_resolved_path():
    """codex/opencode must put the RESOLVED path in argv[0], not the bare name."""
    from harness.providers.codex import CodexProvider
    from harness.providers.opencode import RESTRICTED_AGENT, OpenCodeProvider

    dump = "import json,sys; sys.stderr.write(json.dumps(sys.argv[1:])); print('{}')\n"
    with tempfile.TemporaryDirectory() as d:
        codex_shim = _install_fake_cli(d, "codex", dump)
        oc_shim = _install_fake_cli(d, "opencode", dump)
        old = _with_path(d)
        try:
            cx = CodexProvider()
            assert pathlib.Path(cx._bin).samefile(codex_shim)
            oc = OpenCodeProvider()
            assert pathlib.Path(oc._bin).samefile(oc_shim)
            assert oc._agent == RESTRICTED_AGENT
        finally:
            os.environ["PATH"] = old


def test_cli_provider_round_trips_non_ascii_model_output():
    """The one call that captures the MODEL'S ANSWER must decode it as UTF-8.

    With plain text=True the parent decodes with locale.getencoding() (cp1252 on
    a stock Windows box), so a '>=' or an em dash comes back as mojibake, is
    persisted into llm_trace.jsonl, and poisons every later prompt --- and the
    trace stops being byte-identical across platforms, which is what the
    bit-for-bit replay claim rests on.
    """
    from harness.providers.codex import CodexProvider

    events = json.dumps({"type": "item.completed", "item": {
        "type": "agent_message", "text": _NON_ASCII}}, ensure_ascii=False) + "\n"
    body = ("import sys\n"
            f"sys.stdout.buffer.write({events.encode('utf-8')!r})\n")
    with tempfile.TemporaryDirectory() as d:
        _install_fake_cli(d, "codex", body)
        old = _with_path(d)
        try:
            resp = CodexProvider().complete(
                "system", [{"role": "user", "content": "q"}], model="gpt-5.6")
        finally:
            os.environ["PATH"] = old
    assert resp.text == _NON_ASCII, repr(resp.text)


def test_malformed_bytes_degrade_visibly_instead_of_killing_the_run():
    """An undecodable byte must become U+FFFD, not a UnicodeDecodeError.

    Under the locale codepage some byte sequences raise mid-read and take the
    whole iteration down; errors='replace' keeps the run alive and makes the
    damage visible in the recorded text.
    """
    body = "import sys\nsys.stdout.buffer.write(b'ok \\xff\\xfe tail')\n"
    with tempfile.TemporaryDirectory() as d:
        shim = _install_fake_cli(d, "zora_badbytes_cli", body)
        out = run_cli([str(shim)])
    assert out.startswith("ok ")
    assert "�" in out
    assert "tail" in out
