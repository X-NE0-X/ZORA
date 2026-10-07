"""Provider independence: given the SAME model outputs, two different providers
produce a byte-for-byte identical execution layer (formula / metrics / verdict).
Only the provenance fields (provider / model) may differ. This is what lets a
run be reproduced or audited regardless of which backend generated the factors."""
import pytest


import tempfile

from harness.config import RunConfig
from harness.providers.base import LLMProvider, ProviderResponse
from harness.providers.scripted import DEFAULT_SCRIPT
from harness.runner import _deep_eq, _replay_core, run_once, run_walk_forward
from harness.store import Store

SYMS = ["A", "B", "C", "D", "E", "F"]


class _RelabelProvider(LLMProvider):
    """Emits a fixed script (like the scripted provider) but under a chosen
    provider/model label --- two of these stand in for two different backends
    that happened to return identical text."""

    def __init__(self, name, model, script):
        self.name = name
        self._model = model
        self._script = list(script)
        self._i = 0

    def complete(self, system, messages, *, model, seed=17, max_tokens=2048):
        text = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return ProviderResponse(text=text, model=self._model, provider=self.name)


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=126, min_oos_days=20,
        provider="scripted", seed=17, run_name="test_xprov", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def test_same_outputs_different_providers_match_execution_layer():
    cfg = _cfg()
    prov_a = _RelabelProvider("alpha-backend", "alpha-model-1", DEFAULT_SCRIPT)
    prov_b = _RelabelProvider("beta-backend", "beta-model-9", DEFAULT_SCRIPT)

    # register=False keeps both runs pure (no FactorEngine side effects)
    ra = run_once(cfg, provider=prov_a, register=False)
    rb = run_once(cfg, provider=prov_b, register=False)

    # execution layer is identical...
    assert _deep_eq(_replay_core(ra), _replay_core(rb)), \
        "identical LLM outputs must yield an identical execution layer"
    # ...while the provenance faithfully records which backend produced them
    assert ra["provider"] == "alpha-backend" and rb["provider"] == "beta-backend"
    assert ra["provider"] != rb["provider"]


def test_walk_forward_matches_run_once_replay_core():
    # a recorded walk-forward date and a standalone run_once at the same date,
    # driven by the same outputs, agree on the execution layer
    with tempfile.TemporaryDirectory() as root:
        wf = run_walk_forward(
            _cfg(t_0="2021-01-01", t_p="2021-01-01", run_name="xprov_wf"),
            artifacts_root=root,
        )
        wf_rec = wf["records"][0]

        solo = run_once(_cfg(research_date="2021-01-01"),
                        provider=_RelabelProvider("x", "y", DEFAULT_SCRIPT),
                        register=False)
        assert _deep_eq(_replay_core(wf_rec), _replay_core(solo))

        # sanity: the recorded provider label is the real inner provider's name
        store = Store(root, "xprov_wf")
        assert store.read_manifest()["provider"] == "scripted"
