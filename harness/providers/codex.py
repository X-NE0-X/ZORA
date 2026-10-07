"""OpenAI Codex CLI provider --- ChatGPT-login auth, non-interactive `codex exec`.

Uses the cached ChatGPT sign-in (run ``codex login`` once) rather than an API
key. ``codex exec`` prints ONLY the final message to stdout (logs go to stderr),
runs read-only by default, so it just answers the prompt without editing files.
"""
from __future__ import annotations

import json
import tempfile

from .base import LLMProvider, ProviderResponse, hash_prompt
from .cli_base import ensure_binary, join_prompt, run_cli, scrubbed_env
from .codex_controls import ProviderConfigurationError, validate_controls, validate_model


class CodexProvider(LLMProvider):
    name = "codex"

    def __init__(self, model: str | None = None, sandbox: str = "read-only",
                 use_chatgpt_login: bool = True, timeout: int = 600):
        # Keep the RESOLVED path: on Windows `codex` is an npm .cmd shim that
        # `shutil.which` finds via PATHEXT but CreateProcess cannot launch by
        # bare name (WinError 2) --- see cli_base.ensure_binary.
        self._bin = ensure_binary("codex")
        self._model = model
        self._sandbox = sandbox
        self._login = use_chatgpt_login
        self._timeout = timeout

    def _pick_model(self, model: str | None) -> str:
        if self._model and model and self._model != model:
            raise ProviderConfigurationError(
                f"Codex constructor model {self._model!r} conflicts with requested model {model!r}")
        return validate_model(model or self._model)

    def _child_env(self) -> dict:
        """Environment for the ``codex`` subprocess with the vendor keyring scrubbed.

        With ChatGPT-plan login, keep nothing (Codex uses the cached sign-in);
        otherwise keep only the OpenAI/Codex keys it authenticates with. Every
        other managed vendor credential is dropped so the third-party CLI never
        sees keys it has no use for.
        """
        keep = () if self._login else ("OPENAI_API_KEY", "CODEX_API_KEY")
        return scrubbed_env(keep=keep)

    def complete(self, system, messages, *, model, seed=17, max_tokens=10000,
                 temperature: float | None = None,
                 reasoning_effort: str | None = None,
                 response_format: dict | None = None,
                 structured_schema: dict | None = None,
                 structured_retry_count: int = 2):
        use_model = validate_controls(self._pick_model(model), reasoning_effort)
        prompt = join_prompt(system, messages)
        cmd = [self._bin, "exec", "--skip-git-repo-check",
               "--sandbox", self._sandbox]
        cmd += ["-m", use_model]
        if reasoning_effort is not None:
            cmd += ["-c", f'model_reasoning_effort="{reasoning_effort}"']
        if self._login:
            cmd += ["-c", 'forced_login_method="chatgpt"']
        cmd += [prompt]

        cmd = cmd[:-1]
        cmd += ["--ignore-user-config", "--ephemeral", "--json"]
        for feature in ("shell_tool", "unified_exec", "apps", "plugins", "remote_plugin",
                        "multi_agent", "hooks", "memories", "browser_use", "computer_use",
                        "view_image", "image_generation", "skill_search", "goals"):
            cmd += ["--disable", feature]
        cmd += ["-c", 'web_search="disabled"', "-c", "agents.enabled=false",
                "-c", "mcp_servers={}"]
        with tempfile.TemporaryDirectory(prefix="zora-idea-") as isolated:
            cmd += ["--cd", isolated, prompt]
            events = run_cli(cmd, timeout=self._timeout, env=self._child_env())
        final_messages = []
        for line in events.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProviderConfigurationError("Codex isolation: invalid event stream") from exc
            item = event.get("item", {})
            kind = item.get("type")
            if kind and kind not in ("agent_message", "reasoning"):
                raise ProviderConfigurationError(f"Codex isolation: unexpected item {kind!r}")
            if event.get("type") in ("error", "turn.failed"):
                raise ProviderConfigurationError("Codex isolated completion failed")
            if event.get("type") == "item.completed" and kind == "agent_message":
                final_messages.append(item.get("text", ""))
        if not final_messages:
            raise ProviderConfigurationError("Codex isolation: no final message")
        out = final_messages[-1]
        isolation = "isolated cwd; inherited config/MCP/plugins ignored; tool features disabled; zero observed tool events"
        return ProviderResponse(
            text=out.strip(), model=use_model,
            provider=self.name, prompt_hash=hash_prompt(system, messages),
            meta={"configuration": {
                "requested_model": model,
                "selected_model": use_model,
                "model_selection": "explicit CLI -m (not server-reported)",
                "reasoning_effort": reasoning_effort,
                "reasoning_source": "explicit CLI override" if reasoning_effort else "client default (unverified)",
                "unsupported_llm_controls": ["seed", "temperature", "max_tokens"],
                "tool_isolation": isolation,
            }},
        )
