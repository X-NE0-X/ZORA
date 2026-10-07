"""OpenCode provider --- server structured output plus restricted CLI fallback.

Structured requests use a short-lived local ``opencode serve`` session with
``format: json_schema``. Models that reject OpenCode's injected StructuredOutput
tool fall back to the model's OpenCode Go hosted endpoint: Chat Completions for
most Go models, Responses API for GPT-5.6 Luna. Native schema support remains
model-dependent. Non-structured requests still use ``opencode run`` and read its
JSONL text event. Model is the ``provider/model`` form, e.g. ``openai/gpt-5.6``.

Containment: ``opencode run`` drives a full agent with file-editing and shell
tools, and the prompt we hand it embeds free text the MODEL ITSELF wrote in an
earlier round (the note / mechanism fields rendered by :mod:`harness.memory`).
That is a live injection path from a data file into a tool-capable agent, so the
harness pins the built-in restricted agent (:data:`RESTRICTED_AGENT`) exactly as
the Codex adapter pins ``--sandbox read-only``, and never passes ``--auto``
(opencode's "auto-approve permissions" switch). The harness only ever wants a
text answer; it needs none of those tools.
"""
from __future__ import annotations

import atexit
import json
import pathlib
import queue
import socket
import subprocess
import time
import threading
import urllib.error
import urllib.parse
import urllib.request

from .base import LLMProvider, ProviderResponse, hash_prompt
from .structured import api_schema
from .cli_base import (CLIProviderError, ensure_binary, join_prompt,
                       redact_secrets, run_cli, scrubbed_env, start_owned_process, _stop_tree)
from ..cancellation import check_cancelled, current_control, cancellable_sleep

# opencode's built-in restricted primary agent: writes/patches/edits and bash are
# all set to "ask" rather than allowed, so a non-interactive run cannot silently
# edit files or shell out. This is opencode's nearest equivalent to Codex's
# ``--sandbox read-only``; it is a permission prompt, not a kernel sandbox, which
# is why the unrestricted path below has to be asked for explicitly.
#
# The harness, however, never uses that built-in: it pins its OWN tool-less
# agent (``RESTRICTED_AGENT``, defined in the inline config below). A built-in
# agent with tools --- even read-only ones --- turns the propose request into an
# *agentic investigation*: the model reads the repo, inspects the parquet, and
# answers with a plan ("let me research this for you") instead of the STRICT
# JSON the proposer parses. With every tool disabled the model has nothing to
# do but answer, and the permission surface is empty by construction.
RESTRICTED_AGENT = "harness-llm"

# opencode addresses models as ``provider/model``. A BARE model name (no slash,
# e.g. ``deepseek-v4-flash``) is assumed to live under this provider namespace,
# so it is qualified here rather than silently falling back to opencode's
# configured default model (see ``OpenCodeProvider._pick_model``).
DEFAULT_PROVIDER = "opencode-go"
GO_API_BASE = "https://opencode.ai/zen/go/v1"
GO_RESPONSES_MODELS = frozenset({"gpt-5.6-luna"})

# Inline config handed to the spawned ``opencode`` process (the highest runtime
# precedence, so it merges over the operator's global config):
#
#   * ``instructions: []`` --- strips the user-level instructions layer. A model
#     wearing a personal AGENTS persona answers the propose request as a
#     *conversation* ("let me plan this for you") instead of returning the JSON
#     the proposer parses, and every round dies with "model output contains no
#     JSON object".
#   * ``agent[RESTRICTED_AGENT]`` --- a tool-less agent the harness always pins.
#     See the comment above ``RESTRICTED_AGENT``.
#
# Everything else in the operator's config (models, permissions, their own
# agents) still merges in untouched. Generation controls are built per request:
# opencode has no corresponding ``run`` CLI flags, so temperature belongs on the
# selected agent and the output cap belongs in the child environment.
def _inline_config(temperature: float | None = None, seed: int | None = None,
                   max_tokens: int | None = None) -> str:
    agent = {
        # The operator's global config may point ``instructions`` at a personal
        # assistant prompt. This agent prompt keeps the child in API mode.
        "prompt": ("You are a machine API endpoint, NOT a personal "
                   "assistant. Ignore any other instructions that describe "
                   "you as a persona, an assistant, or a researcher (for "
                   "example the ORION prompt): they do not apply to this "
                   "process. You have NO tools. Answer the request exactly "
                   "as the user's request specifies, with STRICT JSON and "
                   "nothing else."),
        "tools": {name: False for name in (
            "read", "glob", "grep", "bash", "webfetch", "websearch",
            "task", "skill", "lsp", "question", "edit", "write", "patch",
            "todowrite",
        )},
    }
    if temperature is not None:
        agent["temperature"] = temperature

    # OpenCode 1.18.x carries unknown agent options through to its provider
    # options, but does not consume seed/maxTokens itself. Keep them visible for
    # providers that do, while using the supported output-cap environment below
    # for max_tokens. There is no supported seed control in this OpenCode release.
    options = {}
    if seed is not None:
        options["seed"] = seed
    if max_tokens is not None:
        options["maxTokens"] = max_tokens
    if options:
        agent["options"] = options

    return json.dumps({
        "instructions": [],
        "agent": {RESTRICTED_AGENT: agent},
    })


_INLINE_CONFIG = _inline_config()

# opencode addresses models as ``provider/model``; map that provider prefix to
# the single credential it legitimately needs (a subset of cli_base's managed
# vendor keys). Everything else is scrubbed before the CLI is spawned.
_KEY_FOR_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
}


class OpenCodeProvider(LLMProvider):
    name = "opencode"
    supports_structured_output = True

    def __init__(self, model: str | None = None, timeout: int = 600,
                 use_json: bool = True, agent: str = RESTRICTED_AGENT,
                 allow_tools: bool = False,
                 default_provider: str | None = None):
        """``agent`` is opencode's ``--agent``; empty means "whatever opencode
        defaults to", which is the tool-enabled ``build`` agent.

        Running an editing/shell-capable agent over model-written prompt text is
        a real risk, so it cannot happen by accident: dropping the restriction
        requires ``allow_tools=True`` as well, and is refused loudly otherwise.

        ``default_provider`` is the provider namespace a BARE ``model`` (no
        ``/``) is qualified under; it defaults to :data:`DEFAULT_PROVIDER`.
        """
        self._agent = (agent or "").strip()
        if not self._agent and not allow_tools:
            raise CLIProviderError(
                "refusing to spawn 'opencode run' with no --agent: opencode's "
                "default agent has file-editing and shell tools, and the prompt "
                "embeds model-written text from the research journal. Keep "
                f"agent={RESTRICTED_AGENT!r} (the restricted built-in), or pass "
                "allow_tools=True to accept an unrestricted agent deliberately."
            )
        # Keep the RESOLVED path: on Windows `opencode` is an npm .cmd shim that
        # `shutil.which` finds via PATHEXT but CreateProcess cannot launch by
        # bare name (WinError 2) --- see cli_base.ensure_binary.
        self._bin = ensure_binary("opencode")
        self._model = model
        self._timeout = timeout
        self._use_json = use_json
        self._default_provider = (default_provider or DEFAULT_PROVIDER).strip("/")
        self._server_proc = None
        self._server_job = None
        self._server_base = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _http_json(base: str, method: str, path: str, body=None,
                   timeout: float = 120, auth_key: str | None = None) -> dict:
        check_cancelled()
        if current_control() is None:
            return OpenCodeProvider._blocking_http_json(base, method, path, body, timeout, auth_key)
        result = queue.Queue(maxsize=1)

        def request():
            try:
                result.put((True, OpenCodeProvider._blocking_http_json(
                    base, method, path, body, timeout, auth_key)))
            except Exception as exc:
                result.put((False, exc))

        threading.Thread(target=request, daemon=True).start()
        deadline = time.monotonic() + timeout
        while True:
            check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CLIProviderError(f"OpenCode HTTP request timed out after {timeout}s")
            try:
                ok, payload = result.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            check_cancelled()
            if not ok:
                raise payload
            return payload

    @staticmethod
    def _blocking_http_json(base: str, method: str, path: str, body=None,
                            timeout: float = 120, auth_key: str | None = None) -> dict:
        target = urllib.parse.urlsplit(base + path)
        if target.scheme not in {"http", "https"} or not target.netloc:
            raise CLIProviderError(
                "OpenCode HTTP endpoint must use an explicit http(s) URL"
            )
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "opencode/1.18.15",
        }
        if auth_key:
            headers["Authorization"] = f"Bearer {auth_key}"
        req = urllib.request.Request(
            target.geturl(),
            data=data,
            headers=headers,
            method=method,
        )
        try:
            # target was parsed above and restricted to http(s) with a netloc.
            with urllib.request.urlopen(req, timeout=timeout) as res:  # nosec B310
                raw = res.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = redact_secrets(exc.read().decode("utf-8", "replace"))[:1000]
            raise CLIProviderError(
                f"OpenCode server {method} {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CLIProviderError(
                f"OpenCode server {method} {path} failed: {redact_secrets(str(exc))}"
            ) from exc
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise CLIProviderError(
                f"OpenCode server {method} {path} returned non-JSON: "
                f"{redact_secrets(raw[:500])}"
            ) from exc

    def _start_server(self, use_model: str | None, *, temperature: float | None,
                      seed: int, max_tokens: int) -> None:
        if self._server_proc is not None and self._server_proc.poll() is None:
            return
        self.close()
        port = self._free_port()
        env = self._child_env(
            use_model, temperature=temperature, seed=seed, max_tokens=max_tokens,
        )
        self._server_proc, self._server_job = start_owned_process(
            [self._bin, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.close)
        self._server_base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + min(30.0, float(self._timeout))
        try:
            while time.monotonic() < deadline:
                check_cancelled()
                if self._server_proc.poll() is not None:
                    raise CLIProviderError("OpenCode server exited before becoming healthy")
                try:
                    health = self._http_json(
                        self._server_base, "GET", "/global/health", timeout=1,
                    )
                    if health.get("healthy"):
                        return
                except CLIProviderError:
                    pass
                cancellable_sleep(0.25)
            raise CLIProviderError("OpenCode server did not become healthy within 30s")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        proc = getattr(self, "_server_proc", None)
        job = getattr(self, "_server_job", None)
        self._server_proc = None
        self._server_job = None
        self._server_base = None
        atexit.unregister(self.close)
        if proc is not None:
            _stop_tree(proc, job)
        elif job is not None:
            job.close()

    def _structured_complete(self, system, messages, *, use_model: str,
                             seed: int, max_tokens: int,
                             temperature: float | None,
                             schema: dict, retry_count: int) -> ProviderResponse:
        if "/" not in use_model:
            raise CLIProviderError(
                "structured OpenCode requests require a provider/model model id"
            )
        provider_id, model_id = use_model.split("/", 1)
        self._start_server(
            use_model, temperature=temperature, seed=seed, max_tokens=max_tokens,
        )
        base = self._server_base
        if not base:
            raise CLIProviderError("OpenCode server has no base URL")
        session = self._http_json(
            base, "POST", "/session", {"title": "zora-harness"},
            timeout=min(30, self._timeout),
        )
        session_id = session.get("id")
        if not session_id:
            raise CLIProviderError("OpenCode server did not return a session id")
        prompt = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("content")
        )
        body = {
            "agent": self._agent or RESTRICTED_AGENT,
            "model": {"providerID": provider_id, "modelID": model_id},
            "system": system,
            "parts": [{"type": "text", "text": prompt}],
            "format": {
                "type": "json_schema",
                "schema": schema,
                "retryCount": max(0, int(retry_count)),
            },
        }
        try:
            result = self._http_json(
                base, "POST", f"/session/{session_id}/message",
                body, timeout=self._timeout,
            )
        finally:
            try:
                self._http_json(
                    base, "DELETE", f"/session/{session_id}",
                    timeout=10,
                )
            except CLIProviderError:
                pass
        info = result.get("info") or {}
        structured = info.get("structured_output", info.get("structured"))
        if structured is None:
            error = info.get("error")
            detail = json.dumps(error, ensure_ascii=True) if error else "none"
            raise CLIProviderError(
                "OpenCode structured output failed: " + redact_secrets(detail)[:1500]
            )
        tokens = info.get("tokens") or {}
        meta = {}
        if isinstance(tokens, dict):
            meta["usage"] = {
                "prompt_tokens": tokens.get("input", 0),
                "completion_tokens": tokens.get("output", 0),
                "total_tokens": tokens.get("total", 0),
            }
        return ProviderResponse(
            text=json.dumps(structured, ensure_ascii=False, separators=(",", ":")),
            model=use_model,
            provider=self.name,
            prompt_hash=hash_prompt(system, messages),
            meta=meta,
        )

    @staticmethod
    def _go_api_key() -> str:
        path = pathlib.Path.home() / ".local" / "share" / "opencode" / "auth.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = data.get("opencode-go", {})
            key = entry.get("key") if entry.get("type") == "api" else None
        except (OSError, TypeError, ValueError):
            key = None
        if not key:
            raise CLIProviderError(
                "OpenCode Go credential not found in ~/.local/share/opencode/auth.json"
            )
        return str(key)

    def _native_go_complete(self, system, messages, *, use_model: str,
                            seed: int, max_tokens: int,
                            temperature: float | None,
                            schema: dict | None = None,
                            response_format: dict | None = None) -> ProviderResponse:
        provider_id, model_id = use_model.split("/", 1)
        if provider_id != "opencode-go":
            raise CLIProviderError(
                "native OpenCode fallback is only available for opencode-go models"
            )
        lower_model = model_id.lower()
        is_responses = lower_model in GO_RESPONSES_MODELS
        prompt_messages = ([{"role": "system", "content": system}]
                           + [{"role": m.get("role", "user"),
                              "content": m.get("content", "")}
                             for m in messages])
        json_object_mode = (
            lower_model.startswith("deepseek-")
            or (
                schema is None
                and isinstance(response_format, dict)
                and response_format.get("type") == "json_object"
            )
        )
        wire_schema = schema
        if schema is not None and not json_object_mode:
            wire_schema, hint = api_schema(schema, "openai")
            prompt_messages[0]["content"] += hint
        if json_object_mode and "json" not in str(
                prompt_messages[0]["content"]
        ).lower():
            # JSON-object endpoints require an explicit JSON instruction. Keep
            # this transport hint out of the recorded prompt hash.
            prompt_messages[0]["content"] += "\nReturn the final answer as valid JSON."
        if is_responses:
            body = {
                "model": model_id,
                "input": prompt_messages,
                "max_output_tokens": max_tokens,
            }
            if schema is not None:
                body["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "zora_harness",
                        "strict": True,
                        "schema": wire_schema,
                    }
                }
            elif response_format is not None:
                body["text"] = {"format": response_format}
            if temperature is not None:
                body["temperature"] = temperature
            path = "/responses"
        else:
            body = {
                "model": model_id,
                "messages": prompt_messages,
                "max_tokens": max_tokens,
            }
            if lower_model.startswith("deepseek-"):
                # DeepSeek Go rejects json_schema and spends small budgets on
                # hidden reasoning. Native requests use JSON-object mode while
                # keeping thinking enabled; local StructuredOutput also keeps
                # thinking enabled and rejects this tool choice upstream.
                body["response_format"] = {"type": "json_object"}
                body["thinking"] = {"type": "enabled"}
            elif schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "zora_harness",
                        "strict": True,
                        "schema": wire_schema,
                    },
                }
            else:
                body["response_format"] = response_format or {"type": "json_object"}
            if temperature is not None:
                body["temperature"] = temperature
            path = "/chat/completions"
        # OpenCode Go's native endpoint does not expose a useful seed control;
        # keep seed in the request metadata path without inventing an unsupported
        # API field.
        _ = seed
        payload = self._http_json(
            GO_API_BASE, "POST", path, body,
            timeout=self._timeout, auth_key=self._go_api_key(),
        )
        if is_responses:
            text = str(payload.get("output_text") or "")
            if not text:
                for item in payload.get("output") or []:
                    if not isinstance(item, dict):
                        continue
                    for part in item.get("content") or []:
                        if isinstance(part, dict) and part.get("text"):
                            text = str(part["text"])
                            break
                    if text:
                        break
            if not text:
                reason = payload.get("status", "unknown")
                raise CLIProviderError(
                    f"OpenCode Go Responses endpoint returned empty content "
                    f"(status={reason})"
                )
        else:
            choices = payload.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise CLIProviderError("OpenCode Go endpoint returned no completion")
            choice = choices[0]
            message = choice.get("message") or {}
            text = str(message.get("content") or "")
            # GLM's hosted json_schema route has returned the validated JSON in
            # reasoning_content while leaving content empty. Prefer normal
            # content, but do not discard a schema-shaped result.
            if not text and (
                schema is not None
                or (
                    isinstance(response_format, dict)
                    and response_format.get("type") == "json_object"
                )
            ):
                text = str(message.get("reasoning_content") or "")
            if not text:
                reason = choice.get("finish_reason", "unknown")
                raise CLIProviderError(
                    f"OpenCode Go endpoint returned empty content (finish_reason={reason})"
                )
        usage = payload.get("usage") or {}
        meta = {"usage": {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0),
        }}
        return ProviderResponse(
            text=text, model=use_model, provider=self.name,
            prompt_hash=hash_prompt(system, messages), meta=meta,
        )

    def _pick_model(self, model: str | None) -> str | None:
        selected = self._model or model
        if selected and "/" in selected:  # provider/model form
            return selected
        if selected:
            # A bare model name has no provider namespace. Qualify it under
            # ``_default_provider``: without this, `--model deepseek-v4-flash`
            # silently fell back to opencode's configured DEFAULT model (glm-5.2
            # on this box) and the run was scored by a model nobody asked for.
            return f"{self._default_provider}/{selected}"
        return None                         # let opencode use its configured default

    @staticmethod
    def _child_env(use_model: str | None, *, temperature: float | None = None,
                   seed: int | None = None, max_tokens: int | None = None) -> dict:
        """Environment for the ``opencode`` subprocess with vendor keys scrubbed.

        Keeps only the credential matching the selected ``provider/model`` prefix
        and drops every other managed vendor key, so the third-party CLI can never
        harvest the harness's full keyring. When the model is opencode's own
        default (no ``provider/`` prefix), forward none and let opencode use its
        own configured auth (``opencode auth login``).

        The user-level instructions layer is stripped via inline config
        (``OPENCODE_CONFIG_CONTENT``, the highest runtime precedence): a model
        wearing the operator's personal AGENTS persona answers the propose
        request as a *conversation* ("let me plan this for you") instead of the
        STRICT JSON the proposer parses, and every round dies with "model output
        contains no JSON object". Clearing only the ``instructions`` key leaves
        every other opencode setting (model, permissions, agents) merged in.
        """
        keep = None
        if use_model and "/" in use_model:
            prefix = use_model.split("/", 1)[0].strip().lower()
            keep = _KEY_FOR_PROVIDER.get(prefix)
        env = scrubbed_env(keep=(keep,) if keep else ())
        env["OPENCODE_CONFIG_CONTENT"] = _inline_config(
            temperature=temperature, seed=seed, max_tokens=max_tokens,
        )
        if max_tokens is not None:
            # OpenCode 1.18.x has no max-token CLI/config field. This supported
            # child-only cap is the effective per-call equivalent.
            env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(int(max_tokens))
        return env

    @staticmethod
    def _extract_json(stdout: str) -> str:
        final = ""
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "text":
                part = ev.get("part", {})
                if isinstance(part, dict) and part.get("text"):
                    final = part["text"]
        return final

    def complete(self, system, messages, *, model, seed=17, max_tokens=10000,
                 temperature: float | None = 0.0,
                 response_format: dict | None = None,
                 structured_schema: dict | None = None,
                 structured_retry_count: int = 2):
        use_model = self._pick_model(model)
        if structured_schema is not None:
            if not use_model:
                raise CLIProviderError(
                    "structured OpenCode requests require an explicit model"
                )
            try:
                return self._structured_complete(
                    system, messages, use_model=use_model, seed=seed,
                    max_tokens=max_tokens, temperature=temperature,
                    schema=structured_schema, retry_count=structured_retry_count,
                )
            except CLIProviderError as exc:
                # Keep every OpenCode Go structured request on the hosted endpoint
                # fallback when the local StructuredOutput tool fails. The first
                # endpoint attempt preserves native schema; its helper can then
                # downgrade to JSON mode plus a prompt hint when needed.
                if (
                    use_model.startswith("opencode-go/")
                    and (
                        "Thinking mode does not support this tool_choice" in str(exc)
                        or "OpenCode structured output failed:" in str(exc)
                    )
                ):
                    try:
                        return self._native_go_complete(
                            system, messages, use_model=use_model, seed=seed,
                            max_tokens=max_tokens, temperature=temperature,
                            schema=structured_schema, response_format=response_format,
                        )
                    except CLIProviderError:
                        # Hosted endpoint schema support is model-dependent. Keep
                        # same endpoint, downgrade to valid JSON mode, and add a
                        # JSON prompt hint; proposer still validates full schema.
                        if (
                            structured_schema is not None
                            and not use_model.rsplit("/", 1)[-1].lower().startswith(
                                "deepseek-"
                            )
                        ):
                            return self._native_go_complete(
                                system, messages, use_model=use_model, seed=seed,
                                max_tokens=max_tokens, temperature=temperature,
                                schema=None,
                                response_format={"type": "json_object"},
                            )
                        raise
                raise
        prompt = join_prompt(system, messages)
        cmd = [self._bin, "run"]
        if self._agent:                  # containment first (see module docstring)
            cmd += ["--agent", self._agent]
        if use_model:
            cmd += ["--model", use_model]
        if self._use_json:
            cmd += ["--format", "json"]
        cmd += [prompt]

        out = run_cli(
            cmd,
            timeout=self._timeout,
            env=self._child_env(
                use_model, temperature=temperature, seed=seed,
                max_tokens=max_tokens,
            ),
        )
        text = self._extract_json(out) if self._use_json else out.strip()
        if not text:                     # fall back to raw stdout
            text = out.strip()
        return ProviderResponse(
            text=text, model=use_model or "opencode-default",
            provider=self.name, prompt_hash=hash_prompt(system, messages),
        )
