"""Externalised prompts load and render; interpretability toggles correctly.

Also covers the prompt library as a *configuration surface*: the README invites
hand-editing these files, so a typo'd filename must fail as itself (a named
PromptAssetError, before the first iteration) instead of being downgraded to a
nameless "provider error" once per round.
"""
import pytest

import contextlib
import io
import json
import os
import tempfile

from harness import promptlib
from harness.config import RunConfig
from harness.proposer import build_messages, system_prompt
from harness.factor.protocol import STRICT


@contextlib.contextmanager
def _prompts_dir(path: str):
    """Point promptlib at ``path`` for the duration, restoring after.

    The cache is keyed by prompt NAME only, so it is dropped on the way in AND
    on the way out --- otherwise a spec loaded from the fake directory would
    leak into every later test.
    """
    old = promptlib.PROMPTS_DIR
    promptlib.PROMPTS_DIR = path
    promptlib._CACHE.clear()
    try:
        yield
    finally:
        promptlib.PROMPTS_DIR = old
        promptlib._CACHE.clear()


@contextlib.contextmanager
def _patch(obj, name: str, value):
    """Temporarily replace an attribute, restoring after."""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _write(directory: str, name: str, text: str) -> None:
    with open(os.path.join(directory, name + ".json"), "w", encoding="utf-8") as fh:
        fh.write(text)


def test_all_prompts_load():
    names = promptlib.list_prompts()
    assert {"system_propose", "user_initial", "user_refine",
            "interpretability_block"} <= set(names)
    for name in names:
        spec = promptlib.load(name)
        assert "template" in spec


# --- the prompt library as a configuration surface (runtime-02) ------------------
def test_required_prompts_match_the_directory():
    # REQUIRED_PROMPTS is the manifest the readiness gate verifies; it must not
    # drift from the files on disk. A new prompt left off the list would never
    # be checked, and a listed-but-deleted one would fail the gate everywhere.
    assert sorted(promptlib.REQUIRED_PROMPTS) == promptlib.list_prompts()


def test_missing_prompt_raises_a_named_configuration_error():
    try:
        promptlib.load("user_intial")           # the classic typo
    except promptlib.PromptAssetError as exc:
        msg = str(exc)
        assert "user_intial" in msg                  # what was asked for
        assert promptlib.PROMPTS_DIR in msg          # where it was searched
        assert "user_initial" in msg                 # ...and what WAS found
        return
    raise AssertionError("a missing prompt must raise PromptAssetError")


def test_prompt_name_cannot_escape_the_prompt_directory():
    for name in ("../run_config", "..\\run_config", "/tmp/evil", "C:\\evil"):
        try:
            promptlib.load(name)
        except promptlib.PromptAssetError as exc:
            assert "invalid prompt name" in str(exc)
        else:
            raise AssertionError(f"prompt path {name!r} must be rejected")


def test_prompt_error_is_distinguishable_from_a_provider_failure():
    # the runner's per-iteration catch-all downgrades an unrecognised exception
    # to "provider error" and retries it; a broken prompt asset must therefore
    # be a nameable type, not a bare FileNotFoundError indistinguishable from
    # any other IO hiccup inside a provider.
    assert issubclass(promptlib.PromptAssetError, RuntimeError)
    assert not issubclass(promptlib.PromptAssetError, FileNotFoundError)


def test_list_prompts_raises_when_the_directory_is_absent():
    # silently returning [] is exactly how the packaging bug (prompts left out
    # of the wheel) presents -- it must be loud instead.
    missing = os.path.join(tempfile.gettempdir(), "zora_no_such_prompts_dir")
    with _prompts_dir(missing):
        try:
            promptlib.list_prompts()
        except promptlib.PromptAssetError as exc:
            assert "prompt directory not found" in str(exc)
            return
    raise AssertionError("an absent prompt directory must raise, not return []")


def test_malformed_prompt_assets_are_configuration_errors():
    with tempfile.TemporaryDirectory() as d, _prompts_dir(d):
        _write(d, "broken_json", "{not json at all")
        _write(d, "no_template", json.dumps({"name": "x", "description": "y"}))
        _write(d, "empty_template", json.dumps({"template": []}))
        for name, needle in (("broken_json", "not valid JSON"),
                             ("no_template", "'template'"),
                             ("empty_template", "'template'")):
            try:
                promptlib.load(name)
            except promptlib.PromptAssetError as exc:
                assert needle in str(exc), (name, str(exc))
            else:
                raise AssertionError(f"{name} must raise PromptAssetError")


def test_verify_reports_every_broken_asset_in_one_pass():
    with tempfile.TemporaryDirectory() as d, _prompts_dir(d):
        _write(d, "system_propose", json.dumps({"template": ["ok"]}))
        problems = promptlib.verify()
        # one problem per required prompt EXCEPT the single good one
        assert len(problems) == len(promptlib.REQUIRED_PROMPTS) - 1
        assert all("not found" in p for p in problems)
        assert not any("system_propose" in p.split(";")[0] for p in problems)
    assert promptlib.verify() == []          # the real library is intact


def test_broken_prompt_library_blocks_the_run_before_the_first_iteration():
    # runtime-02: a missing prompt used to reach the runner, be downgraded to
    # "provider error: FileNotFoundError: ...user_initial.json" once per
    # iteration, and end the run with "no valid active factor proposal was
    # produced". The readiness gate must stop it before any of that.
    from harness import cli, preflight
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as d, _prompts_dir(d), \
            _patch(preflight, "_probe_engine", lambda: ""), \
            contextlib.redirect_stdout(buf):
        code = cli.main(["run", "--run-name", "prompt_gate_test"])
    out = buf.getvalue()
    assert code == 3                                   # blocked, not attempted
    assert "prompt asset" in out
    assert "provider error" not in out
    assert "no valid active factor" not in out


def test_render_substitutes_everything():
    txt = promptlib.render(
        "system_propose", fields="close, volume", operators="rank, ts_mean",
        objective_label="Sortino", cmp=">=", pass_line="1",
        goodness_dir="higher is better", interpretability="",
        output_format='{"formula": "rank(close)"}',
    )
    assert "close, volume" in txt and "rank, ts_mean" in txt and "Sortino" in txt
    assert "$" not in txt.replace("$placeholder", "")   # no leftover $tokens


def test_interpretability_prompt_respects_contract():
    off = system_prompt(RunConfig(require_interpretability=False))
    on = system_prompt(RunConfig(require_interpretability=True))
    assert "INTERPRETABILITY REQUIRED" in on
    assert "INTERPRETABILITY REQUIRED" not in off
    assert "economic HYPOTHESIS" in on
    assert "Local software controls admission and scoring" in on


def test_research_discipline_constraints_present():
    # the three anti-overfitting constraints must reach the model on every
    # propose call (they live in the always-sent system prompt), independent of
    # the interpretability toggle.
    for interp in (False, True):
        s = system_prompt(RunConfig(require_interpretability=interp))
        assert "At most TWO bindings total" in s
        assert "bounded candidate proposer" in s
        assert "no tool or research-execution authority" in s
        assert "parameters" in s


def test_objective_shapes_prompt():
    cfg_vol = RunConfig(objective="ann_vol", pass_line=0.2)
    s_vol = system_prompt(cfg_vol) + "\n" + build_messages(cfg_vol, None)[0]["content"]
    assert "AnnVol" in s_vol and "<=" in s_vol        # lower is better
    cfg_srt = RunConfig(objective="sortino", pass_line=1.0)
    s_srt = system_prompt(cfg_srt) + "\n" + build_messages(cfg_srt, None)[0]["content"]
    assert "Sortino" in s_srt and ">=" in s_srt


def test_refine_message_includes_history():
    cfg = RunConfig()
    hist = [{"formula": "rank(close)",
             "is_metrics": {"sortino": 0.5, "sharpe": 0.4, "avg_turnover": 0.1}}]
    msgs = build_messages(cfg, hist)
    assert "rank(close)" in msgs[0]["content"]


def test_refine_feeds_full_metrics_and_names_objective():
    # the refine turn hands the model the WHOLE in-sample metric set (no single
    # metric pre-labelled "the score" in the block itself) AND names its
    # optimisation objective + Pass Line, so the agent knows what it is improving
    # while still seeing the full risk/return/activity picture.
    cfg = RunConfig(objective="sortino")
    m = {"sortino": 1.2, "sharpe": 0.9, "calmar": 0.5, "cagr": 0.11,
         "ann_return": 0.11, "ann_vol": 0.2, "maxdd": -0.15, "hit_rate": 0.53,
         "avg_turnover": 0.3, "avg_gross": 1.0, "n": 252}
    content = build_messages(cfg, [{"formula": "-delta(close, 5)",
                                    "is_metrics": m}])[0]["content"]
    for tok in ("sortino=1.2000", "sharpe=0.9000", "calmar=0.5000",
                "maxdd=-0.1500", "hit_rate=0.5300", "avg_gross=1.0000", "n=252"):
        assert tok in content, f"missing metric {tok!r}"
    # the objective IS named now (the agent must know its goal), with the Pass Line
    # and direction --- sortino is higher-is-better.
    assert "Sortino" in content
    assert ">=" in content
    # ...and it tracks the configured objective, not a hardcoded one
    vol = build_messages(RunConfig(objective="ann_vol", pass_line=0.2),
                         [{"formula": "close", "is_metrics": m}])[0]["content"]
    assert "AnnVol" in vol and "<=" in vol       # lower-is-better objective
    # a NaN metric renders as n/a, never a crash or a fake 0.0000
    m2 = {**m, "calmar": float("nan")}
    c2 = build_messages(cfg, [{"formula": "close", "is_metrics": m2}])[0]["content"]
    assert "calmar=n/a" in c2
