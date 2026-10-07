"""Provider hardening:

  * opencode (a third-party CLI) must only ever inherit the ONE vendor key that
    matches the model it is routing to --- never the harness's whole keyring (#10);
  * and, more strongly, the child environment is an ALLOWLIST: an unknown secret
    the harness never heard of must not be forwarded either (security-01);
  * a failing CLI's stderr is quoted into an error that gets PERSISTED into
    llm_trace.jsonl, so credential shapes are redacted first (security-07);
  * ``opencode run`` drives a tool-capable agent over model-written prompt text,
    so it is pinned to opencode's restricted agent, and the unrestricted path
    must be opted into explicitly (security-02);
  * live API providers pin ``temperature=0`` for best-effort determinism, and the
    OpenAI adapter drops a rejected optional param instead of failing (#12).

All offline: the CLI env logic is a pure staticmethod, and the API adapters are
driven with fake clients (no SDK, no key, no network)."""
import json
import os
import sys
import tempfile
import types

from harness.config import RunConfig
from harness.providers.base import (LLMProvider, ProviderResponse,
                                    provider_lifecycle)
from harness.providers.replay import RecordingProvider
from harness.store import Store
from harness.providers import cli_base
from harness.providers.claude import ClaudeProvider
from harness.providers.cli_base import MANAGED_VENDOR_KEYS as _MANAGED_KEYS
from harness.providers.cli_base import (CLIProviderError, redact_secrets,
                                        scrubbed_env)
from harness.providers.codex import CodexProvider
from harness.providers.openai_api import DeepSeekProvider, OpenAIProvider
from harness.providers.cli_base import CLIProviderError
from harness.providers.opencode import RESTRICTED_AGENT, OpenCodeProvider


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --- #10: opencode credential scrubbing ------------------------------------
def test_opencode_forwards_only_matching_provider_key():
    saved = {k: os.environ.get(k) for k in _MANAGED_KEYS}
    try:
        for k in _MANAGED_KEYS:
            os.environ[k] = f"secret-{k}"

        # openai/model -> only OPENAI_API_KEY survives
        env = OpenCodeProvider._child_env("openai/gpt-5.6")
        assert env["OPENAI_API_KEY"] == "secret-OPENAI_API_KEY"
        assert "ANTHROPIC_API_KEY" not in env
        assert "DEEPSEEK_API_KEY" not in env
        assert env.get("PATH"), "the CLI still needs PATH to find its own tools"

        # anthropic/model -> only ANTHROPIC_API_KEY survives
        env2 = OpenCodeProvider._child_env("anthropic/claude-opus-4-8")
        assert env2["ANTHROPIC_API_KEY"] == "secret-ANTHROPIC_API_KEY"
        assert "OPENAI_API_KEY" not in env2

        # opencode default model (no provider prefix) -> forward NO managed key
        for m in (None, "", "gpt-5", "some-bare-model"):
            envd = OpenCodeProvider._child_env(m)
            assert not any(k in envd for k in _MANAGED_KEYS), \
                f"no vendor key should leak for model {m!r}"

        # an unknown provider prefix also forwards nothing (fail closed)
        envu = OpenCodeProvider._child_env("mysteryvendor/model")
        assert not any(k in envu for k in _MANAGED_KEYS)
    finally:
        _restore(saved)


# --- security-01: the child env is an ALLOWLIST, not a blacklist ------------
def test_scrubbed_env_is_an_allowlist_not_a_blacklist():
    """A secret the harness has never heard of must not reach a spawned CLI.

    The old blacklist popped MANAGED_VENDOR_KEYS and forwarded everything else,
    so an unrelated credential in the operator's shell (another tool's token, a
    CI secret, an SSO cookie) went straight to a third-party binary.
    """
    strangers = {
        "ACME_INTERNAL_TOKEN": "super-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "GITHUB_TOKEN": "gh-secret",
        "DATABASE_URL": "postgres://user:pw@host/db",
    }
    saved = {k: os.environ.get(k) for k in strangers}
    saved["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY")
    try:
        os.environ.update(strangers)
        os.environ["HTTPS_PROXY"] = "http://proxy.local:8080"
        env = scrubbed_env()
        for name in strangers:
            assert name not in env, f"{name} must not be forwarded to a sub-CLI"
        # ...while the operational vars a CLI genuinely needs still are
        assert env.get("PATH")
        assert env["HTTPS_PROXY"] == "http://proxy.local:8080"
        assert any(k in env for k in ("HOME", "USERPROFILE")), \
            "a CLI needs a home dir to find its own cached login"
    finally:
        _restore(saved)


def test_scrubbed_env_keep_readmits_exactly_one_credential():
    saved = {k: os.environ.get(k) for k in _MANAGED_KEYS}
    try:
        for k in _MANAGED_KEYS:
            os.environ[k] = f"secret-{k}"
        env = scrubbed_env(keep=("ANTHROPIC_API_KEY",))
        assert env["ANTHROPIC_API_KEY"] == "secret-ANTHROPIC_API_KEY"
        assert [k for k in _MANAGED_KEYS if k in env] == ["ANTHROPIC_API_KEY"]
        # an empty / falsy keep entry is ignored, it does not admit everything
        assert not any(k in scrubbed_env(keep=("", None)) for k in _MANAGED_KEYS)
    finally:
        _restore(saved)


# --- security-07: nothing credential-shaped reaches the persisted trace -----
def test_redact_secrets_masks_credential_shapes():
    text = ("auth failed: OPENAI_API_KEY=sk-proj-ABCDEFGH1234567890 rejected; "
            "Authorization: Bearer abcdefghijklmnop; "
            'config {"api_key": "ghp_ABCDEFGHIJKLMNOPQRST"}')
    out = redact_secrets(text)
    for secret in ("sk-proj-ABCDEFGH1234567890", "abcdefghijklmnop",
                   "ghp_ABCDEFGHIJKLMNOPQRST"):
        assert secret not in out, f"{secret!r} survived redaction"
    assert "<redacted>" in out


def test_redact_secrets_masks_a_live_vendor_key_value():
    saved = {"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY")}
    try:
        # deliberately NOT credential-shaped: only the exact-value pass catches it
        os.environ["ANTHROPIC_API_KEY"] = "zzzq-not-a-known-shape-1234"
        out = redact_secrets("child said: zzzq-not-a-known-shape-1234 is invalid")
        assert "zzzq-not-a-known-shape-1234" not in out
        assert "ANTHROPIC_API_KEY redacted" in out
    finally:
        _restore(saved)


def test_recording_masks_success_and_exception_before_persistence():
    secret = "plain-audit-secret-value-12345"
    saved = {"AUDIT_FAKE_TOKEN": os.environ.get("AUDIT_FAKE_TOKEN")}

    class Echo(LLMProvider):
        name = "echo"

        def __init__(self, fail=False):
            self.fail = fail

        def complete(self, system, messages, **kwargs):
            if self.fail:
                raise RuntimeError(f"provider leaked {secret}")
            return ProviderResponse(text=f"model echoed {secret}", model="m",
                                    provider=self.name)

    try:
        os.environ["AUDIT_FAKE_TOKEN"] = secret
        with tempfile.TemporaryDirectory() as root:
            store = Store(root, "redacted")
            ok = RecordingProvider(Echo(), on_record=store.append_trace)
            response = ok.complete("s", [], model="m")
            assert secret in response.text  # persistence masking does not alter use
            bad = RecordingProvider(Echo(fail=True), on_record=store.append_trace)
            try:
                bad.complete("s", [], model="m")
            except RuntimeError:
                pass
            else:
                raise AssertionError("provider failure must propagate")
            raw = open(store.trace_path, encoding="utf-8").read()
            assert secret not in raw
            assert "redacted" in raw
    finally:
        _restore(saved)


def test_redact_secrets_keeps_the_words_preflight_matches_on():
    """Redaction must not eat the vocabulary the auth-error detector keys off.

    preflight.looks_like_auth_error scans for 'expired', 'unauthorized',
    'api key', ... If redaction swallowed the word after 'token', an auth failure
    would stop being recognised as one and the setup guide would never fire.
    """
    from harness import preflight

    msg = "401 Unauthorized: authentication token expired, invalid key"
    out = redact_secrets(msg)
    assert out == msg                       # nothing credential-shaped in it
    assert preflight.looks_like_auth_error(out)


def test_run_cli_error_message_is_redacted():
    """The message run_cli raises IS what RecordingProvider writes to the trace."""
    import pathlib
    import sys
    import tempfile

    # no pytest fixture: tests/run_tests.py calls every test_* with no arguments
    with tempfile.TemporaryDirectory() as d:
        child = pathlib.Path(d) / "leaky.py"
        child.write_text(
            "import sys\n"
            "sys.stderr.write('boom: api_key=sk-live-ABCDEFGH12345678\\n')\n"
            "sys.exit(3)\n",
            encoding="utf-8",
        )
        try:
            cli_base.run_cli([sys.executable, str(child)])
        except CLIProviderError as exc:
            assert "sk-live-ABCDEFGH12345678" not in str(exc)
            assert "<redacted>" in str(exc)
        else:
            raise AssertionError("a non-zero exit must raise CLIProviderError")


# --- security-02: opencode is not spawned as a tool-capable agent -----------
def test_opencode_pins_the_restricted_agent():
    prov = object.__new__(OpenCodeProvider)     # bypass __init__ (no binary here)
    prov._bin = "/fake/opencode"
    prov._agent = RESTRICTED_AGENT
    prov._model = None
    prov._timeout = 1
    prov._use_json = True

    seen = {}

    def fake_run_cli(cmd, timeout=600, env=None, stdin=None):
        seen["cmd"] = cmd
        return '{"type": "text", "part": {"text": "answer"}}'

    real = cli_base.run_cli
    import harness.providers.opencode as oc
    oc.run_cli = fake_run_cli
    try:
        resp = prov.complete("sys", [{"role": "user", "content": "q"}],
                             model="openai/gpt-5.6")
    finally:
        oc.run_cli = real
    assert resp.text == "answer"
    cmd = seen["cmd"]
    assert cmd[:4] == ["/fake/opencode", "run", "--agent", RESTRICTED_AGENT], cmd
    assert "--auto" not in cmd, \
        "'--auto' auto-approves opencode's permissions -- never pass it"


def test_opencode_refuses_an_unrestricted_agent_without_opt_in():
    # constructing with no agent must fail LOUDLY before the binary is even
    # looked up, so 'opencode not installed' cannot mask the security decision
    try:
        OpenCodeProvider(agent="")
    except CLIProviderError as exc:
        assert "allow_tools" in str(exc), str(exc)
    else:
        raise AssertionError("an unrestricted opencode agent must be refused")


def test_opencode_injects_generation_controls_into_child_config():
    env = OpenCodeProvider._child_env(
        "opencode-go/deepseek-v4-flash", temperature=0.25, seed=7,
        max_tokens=321,
    )
    cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    agent = cfg["agent"][RESTRICTED_AGENT]
    assert agent["temperature"] == 0.25
    assert agent["options"] == {"seed": 7, "maxTokens": 321}
    assert env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "321"


def test_opencode_qualifies_constructor_bare_model():
    prov = object.__new__(OpenCodeProvider)
    prov._model = "deepseek-v4-flash"
    prov._default_provider = "opencode-go"
    assert prov._pick_model(None) == "opencode-go/deepseek-v4-flash"


def test_opencode_native_deepseek_enables_thinking():
    prov = object.__new__(OpenCodeProvider)
    prov._timeout = 1
    seen = {}

    def fake_http(base, method, path, body=None, timeout=120.0, auth_key=None):
        seen["body"] = body
        return {
            "choices": [{
                "message": {"content": '{"ok":true}'},
                "finish_reason": "stop",
            }],
            "usage": {},
        }

    prov._http_json = fake_http
    prov._go_api_key = lambda: "test-key"
    resp = prov._native_go_complete(
        "system", [{"role": "user", "content": "question"}],
        use_model="opencode-go/deepseek-v4-flash", seed=3,
        max_tokens=2048, temperature=0.0,
    )
    assert resp.text == '{"ok":true}'
    assert seen["body"]["thinking"] == {"type": "enabled"}
    assert "JSON" in seen["body"]["messages"][0]["content"]


def test_opencode_http_transport_rejects_non_network_schemes():
    try:
        OpenCodeProvider._http_json("file:///tmp", "GET", "/secret")
    except CLIProviderError as exc:
        assert "http(s)" in str(exc)
    else:
        raise AssertionError("file: endpoints must be rejected before urlopen")


def test_opencode_native_chat_schema_and_reasoning_content_fallback():
    prov = object.__new__(OpenCodeProvider)
    prov._timeout = 1
    seen = {}

    def fake_http(base, method, path, body=None, timeout=120.0, auth_key=None):
        seen.update(method=method, path=path, body=body)
        return {
            "choices": [{
                "message": {"content": "", "reasoning_content": '{"ok":true}'},
                "finish_reason": "stop",
            }],
            "usage": {},
        }

    prov._http_json = fake_http
    prov._go_api_key = lambda: "test-key"
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    resp = prov._native_go_complete(
        "system", [{"role": "user", "content": "question"}],
        use_model="opencode-go/glm-5.2", seed=3, max_tokens=128,
        temperature=0.0, schema=schema,
    )
    assert resp.text == '{"ok":true}'
    assert seen["path"] == "/chat/completions"
    assert seen["body"]["response_format"]["type"] == "json_schema"


def test_opencode_native_luna_uses_responses_schema():
    prov = object.__new__(OpenCodeProvider)
    prov._timeout = 1
    seen = {}

    def fake_http(base, method, path, body=None, timeout=120.0, auth_key=None):
        seen.update(method=method, path=path, body=body)
        return {
            "output": [{
                "content": [{"type": "output_text", "text": '{"ok":true}'}]
            }],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }

    prov._http_json = fake_http
    prov._go_api_key = lambda: "test-key"
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    resp = prov._native_go_complete(
        "system", [{"role": "user", "content": "question"}],
        use_model="opencode-go/gpt-5.6-luna", seed=3, max_tokens=128,
        temperature=0.0, schema=schema,
    )
    assert resp.text == '{"ok":true}'
    assert seen["path"] == "/responses"
    assert "messages" not in seen["body"]
    assert seen["body"]["max_output_tokens"] == 128
    assert seen["body"]["text"]["format"]["type"] == "json_schema"


def test_opencode_falls_from_endpoint_schema_to_json_prompt_mode():
    prov = object.__new__(OpenCodeProvider)
    prov._model = None
    calls = []
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    def structured(*args, **kwargs):
        raise CLIProviderError("OpenCode structured output failed: schema tool error")

    def native(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise CLIProviderError("hosted endpoint rejected json_schema")
        return types.SimpleNamespace(text='{"ok":true}')

    prov._structured_complete = structured
    prov._native_go_complete = native
    resp = prov.complete(
        "system", [{"role": "user", "content": "q"}],
        model="opencode-go/kimi-k3", structured_schema=schema,
    )
    assert resp.text == '{"ok":true}'
    assert calls[0]["schema"] == schema
    assert calls[1]["schema"] is None
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_codex_scrubs_whole_keyring():
    saved = {k: os.environ.get(k) for k in _MANAGED_KEYS}
    try:
        for k in _MANAGED_KEYS:
            os.environ[k] = f"secret-{k}"

        # ChatGPT-login: Codex uses cached sign-in -> forward NO vendor key
        login = object.__new__(CodexProvider)
        login._login = True
        env_login = login._child_env()
        assert not any(k in env_login for k in _MANAGED_KEYS), \
            "chatgpt-login codex must not inherit any vendor key"

        # API-key auth: keep only OpenAI/Codex keys, drop the other eight
        apikey = object.__new__(CodexProvider)
        apikey._login = False
        env_api = apikey._child_env()
        assert env_api["OPENAI_API_KEY"] == "secret-OPENAI_API_KEY"
        assert env_api["CODEX_API_KEY"] == "secret-CODEX_API_KEY"
        for leaked in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
                       "OPENROUTER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY",
                       "XAI_API_KEY", "GOOGLE_API_KEY"):
            assert leaked not in env_api, f"{leaked} must not leak to codex"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- #12: temperature pinning + graceful degradation -----------------------
def _fake_openai_client(text="ok", reject=()):
    reject = set(reject)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        bad = reject & set(kwargs)
        if bad:
            raise RuntimeError(f"Unsupported parameter: {sorted(bad)[0]}")
        msg = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    return client, calls


def _openai_provider(client):
    prov = object.__new__(OpenAIProvider)   # bypass __init__ (no SDK/key needed)
    prov._client = client
    prov._default_model = "gpt-x"
    return prov


def test_openai_pins_temperature_and_seed():
    client, calls = _fake_openai_client(text="hi")
    out = _openai_provider(client)._chat("gpt-x", [{"role": "user", "content": "q"}],
                                         max_tokens=100, seed=17)
    assert out[0] == "hi"
    last = calls[-1]
    assert last["temperature"] == 0
    assert last["seed"] == 17
    assert last["max_tokens"] == 100


def test_openai_drops_rejected_temperature_and_retries():
    client, calls = _fake_openai_client(text="hi", reject={"temperature"})
    out = _openai_provider(client)._chat("o5", [{"role": "user", "content": "q"}],
                                         max_tokens=50, seed=7)
    assert out[0] == "hi"                       # succeeded despite the rejection
    assert len(calls) >= 2                    # it retried
    assert "temperature" not in calls[-1]     # ...without the offending param
    assert calls[-1]["seed"] == 7             # kept the params that were fine


def test_openai_passes_response_format():
    client, calls = _fake_openai_client(text='{"ok":true}')
    fmt = {"type": "json_object"}
    out = _openai_provider(client)._chat(
        "gpt-x", [{"role": "user", "content": "q"}],
        max_tokens=50, seed=1, response_format=fmt,
    )
    assert out[0] == '{"ok":true}'
    assert calls[-1]["response_format"] == fmt


def test_openai_translates_structured_schema_to_json_schema():
    client, calls = _fake_openai_client(text='{"ok":true}')
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    out = _openai_provider(client).complete(
        "system", [{"role": "user", "content": "q"}],
        model="gpt-x", max_tokens=50, seed=1, structured_schema=schema,
    )
    assert out.text == '{"ok":true}'
    assert calls[-1]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "zora_harness", "strict": True, "schema": schema,
        },
    }


def test_deepseek_keeps_json_object_and_injects_json_prompt_hint():
    client, calls = _fake_openai_client(text='{"ok":true}')
    prov = object.__new__(DeepSeekProvider)
    prov._client = client
    prov._default_model = "deepseek-v4-flash"
    out = prov.complete(
        "Return an object.", [{"role": "user", "content": "q"}],
        model="deepseek-v4-flash", max_tokens=50, seed=1,
        response_format={"type": "json_object"},
    )
    assert out.text == '{"ok":true}'
    assert calls[-1]["response_format"] == {"type": "json_object"}
    assert "json" in calls[-1]["messages"][0]["content"].lower()
    assert calls[-1]["extra_body"] == {"thinking": {"type": "enabled"}}


def test_claude_passes_structured_schema_as_output_config():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text='{"ok":true}')
        return types.SimpleNamespace(
            content=[block], stop_reason="end_turn", usage=None,
        )

    prov = object.__new__(ClaudeProvider)
    prov._client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=create),
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    resp = prov.complete(
        "system", [{"role": "user", "content": "q"}],
        model="claude-opus-5", structured_schema=schema,
    )
    assert resp.text == '{"ok":true}'
    assert calls[-1]["output_config"] == {
        "format": {"type": "json_schema", "schema": schema},
    }


def test_openai_handles_token_param_rename():
    client, calls = _fake_openai_client(text="hi", reject={"max_tokens"})
    out = _openai_provider(client)._chat("gpt-x", [{"role": "user", "content": "q"}],
                                         max_tokens=64, seed=1)
    assert out[0] == "hi"
    assert "max_completion_tokens" in calls[-1] and "max_tokens" not in calls[-1]
    assert calls[-1]["max_completion_tokens"] == 64


def test_claude_pins_temperature_and_omits_seed():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text="claude-hi")
        return types.SimpleNamespace(content=[block], stop_reason="end_turn")

    prov = object.__new__(ClaudeProvider)
    prov._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    resp = prov.complete("system", [{"role": "user", "content": "q"}],
                         model="claude-opus-4-8", seed=17)
    assert resp.text == "claude-hi"
    assert calls[-1]["temperature"] == 0
    # Anthropic's Messages API has no seed param -> we must NOT pass one
    assert "seed" not in calls[-1]


def test_api_sdk_clients_receive_explicit_timeouts_and_close():
    seen = {}

    class FakeClient:
        def __init__(self, kind, **kwargs):
            seen[kind] = kwargs
            self.closed = False

        def close(self):
            self.closed = True

    old_openai = sys.modules.get("openai")
    old_anthropic = sys.modules.get("anthropic")
    sys.modules["openai"] = types.SimpleNamespace(
        OpenAI=lambda **kw: FakeClient("openai", **kw))
    sys.modules["anthropic"] = types.SimpleNamespace(
        Anthropic=lambda **kw: FakeClient("anthropic", **kw))
    try:
        op = OpenAIProvider(api_key="test", timeout=17)
        cl = ClaudeProvider(api_key="test", timeout=19)
        assert seen["openai"]["timeout"] == 17.0
        assert seen["anthropic"]["timeout"] == 19.0
        op.close()
        cl.close()
        assert op._client.closed and cl._client.closed
    finally:
        if old_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = old_openai
        if old_anthropic is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = old_anthropic


def test_run_once_closes_only_the_provider_it_constructs():
    import harness.runner as runner

    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    created = FakeProvider()
    supplied = FakeProvider()
    original_get = runner.get_provider
    original_inner = runner._run_once_owned
    runner.get_provider = lambda _name: created
    runner._run_once_owned = lambda *_args, **_kwargs: {"ok": True}
    try:
        assert runner.run_once(RunConfig(provider="scripted")) == {"ok": True}
        assert created.closed
        assert runner.run_once(
            RunConfig(provider="scripted"), provider=supplied
        ) == {"ok": True}
        assert not supplied.closed, "caller-owned providers must remain reusable"
    finally:
        runner.get_provider = original_get
        runner._run_once_owned = original_inner


def test_provider_lifecycle_closes_opencode_at_run_boundary():
    prov = object.__new__(OpenCodeProvider)
    prov._server_proc = None
    prov._server_base = "http://127.0.0.1:1"
    with provider_lifecycle(prov):
        pass
    assert prov._server_proc is None and prov._server_base is None
