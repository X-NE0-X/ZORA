# ZORA Harness

Source-available software. Copyright 2026 X-NE0-X. The combined [LICENSE](LICENSE)
uses Apache License 2.0 terms **subject to Commons Clause v1.0**, which restricts
"Sell" as defined there. Internal company use is allowed; this is not a blanket
ban on every commercial activity. This combination is not OSI-approved open
source and must not be labeled Apache-2.0 alone. Dependencies retain their own
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The validated runtime is Windows / CPython 3.14 with `requirements.txt`.
Other Python versions in the declared range have syntax-only CI jobs, not full
engine or installed-wheel acceptance. See [RELEASE.md](RELEASE.md) for release
gates and [SECURITY.md](SECURITY.md) for private vulnerability reporting.

A small, self-contained, **provider-agnostic** harness that lets an LLM *invent*
alpha factors and puts them through an honest test:

```
natural language -> factor formula -> backtest -> iterate -> Pass / Fail
```

The model writes a factor as a **math formula** composed of general operators
(`rank`, `ts_mean`, `delta`, `ts_corr`, arithmetic, ...). The harness compiles
the formula to a safe AST (no `eval`, whitelist only), backtests it on real
OHLCV, feeds the metrics back so the model can refine it on an **in-sample**
window, and then scores the best formula **once** on a held-out **out-of-sample**
window. The OOS objective clears the **Pass Line** (and the sign is consistent)
→ **Pass**.

## Strict mathematical admission (only supported contract)

New runs use `math_contract = "strict-math-v2"`. The model is a bounded idea
proposer. The program owns mathematical admission, evaluation and verdicts.
`tri_align = false` disables annotations only. Mathematical checks are always enforced.

Candidates have exactly `formula`, `parameters`, `rationale`, `mechanism` and
`expected_sign`. An example is:

```json
{"formula":"-rank(delta(close,w))","parameters":{"w":{"type":"Window","value":20}},"rationale":"Recent price changes may reverse when short-lived demand pressure dissipates.","mechanism":"The negative rank of the own-price lag difference expresses a reversal hypothesis.","expected_sign":1}
```

The whole formula may reference at most two distinct numerical parameter IDs.
Shared IDs count once; equal values under different IDs still count separately.
Windows must be positive JSON integers. Anonymous literals, scalar arithmetic,
unbound parameters, extra/duplicate JSON keys and independent signal mixing are
rejected before scoring. Parameter IDs are alpha-normalized for expression
identity without merging different IDs or changing floating-point order.

The language allows unary transforms, compatible raw spreads, raw price-family
ratios, raw-observable correlation/covariance, and exact own-baseline/own-scale
constructions. Matching a field set or writing a persuasive mechanism does not
establish subject identity. `**` and `pow` mean ordinary real power;
`signed_pow` explicitly means `sign(x)*abs(x)**p`; `signed_log1p` explicitly
means `sign(x)*log(1+abs(x))`. There is no strict `log` alias. The versioned
`harness/factor/contract.py` registry supplies signatures, semantics, mask rules
and the sole execution kernels. There is no historical evaluator or hybrid-power path.

Every evaluated node checks axes and nonfinite output. Source missingness,
warmup, zero denominators/variance and real-power domains have declared masks.
Overflow hidden beneath `rank` is rejected. Evaluation requires a nonempty
valid scope and rejects globally constant signals. No new arbitrary coverage
percentage is imposed; the existing live-exposure floors still govern economic
eligibility. IS evaluation truncates the input at the IS endpoint. Frozen OOS
numerical failure produces `NUMERICAL_REJECT` without changing the frozen candidate or re-asking the model. This rejects that date's factor; later independent research dates can continue.

Static mathematical violations (grammar, units, construction or parameter budget)
fail before scoring, regardless of IS/OOS. Numerical failures depend on observed
values and are checked separately on causal IS and frozen OOS. Testing OOS before
candidate selection would contaminate the held-out result.

Strict VWAP requires a source-provided field; HLC3 remains `typical_price`.
Records separate mathematical admission from economic PASS/FAIL, retain original
bindings and normalized identity, and include distinct IS/OOS numerical records.
The manifest freezes the policy hash. Strict replay refuses policy drift rather
than silently substituting new semantics. A mathematical admission record does
not prove a unique economic cause, predictive power or global originality.

Strict single-date CLI/TUI runs persist each completion as well as walk-forward
runs. Their manifest identifies `execution_mode = "single-date"`, so replay does
not interpret unused walk-forward date controls as additional research dates.
Replay honors the recorded worker setting and supports explicit process/serial
parity checks without new model calls.
Worker thread limits are installed before spawn reloads the main module, as
well as in the initializer; the caller's original environment is restored after
submission, including on failure.

Only the complete current recorded schema and policy hash may replay or resume.
`legacy-v1`, missing contract fields, removed configuration keys and stale policies
fail explicitly. There is no automatic migration or historical compatibility mode.
Existing historical files are retained; unsupported records are never generation
feedback. Current journal feedback requires an explicit recorded OOS close date.
Formula examples below describe mathematical shapes; executable expressions must
use explicit named bindings as shown above.

For ChatGPT OAuth Codex calls, strict mode uses an isolated working directory,
ignores inherited configuration, disables tool features, and rejects unexpected
tool events. Auth remains the existing cached sign-in. Selected model/effort are
recorded as explicit client requests, not server-confirmed identity. Other real
provider capability-isolation paths require their own live acceptance.

`python tests/run_tests.py` is the canonical offline regression entry point. It
runs only current strict-math-v2 cases once each. `python -m pytest -q` uses
the same current-only collection. Use `python tests/run_tests.py --workers 8`
for disjoint file shards; pytest selectors are scoped checks, not additional
full-run passes. Each case has one contract classification; scoped checks are not added to full-run
totals. JSON receipts and JUnit XML are written under `artifacts/regression/`;
the parallel summary rejects overlapping, missing or unfinished cases and a
changed source fingerprint. Collection-only receipts are inventories, not passes.
An explicit `--report-dir` must be empty; existing evidence is never overwritten.
Parallel full runs reject inherited pytest selection filters as well.
The canonical runner creates a fresh repository-local temporary directory for
each invocation. When calling pytest directly on a restricted Windows host,
supply `--basetemp artifacts/<unique-test-directory>`.

> Everything this tool produces is a **hypothetical backtest on historical
> data**. It is not investment advice, and a **Pass** is a research signal — one
> factor survived one held-out window — not a recommendation to trade anything.
> Read [Known biases and limits](#known-biases-and-limits) before you believe a
> number in here.

The heavy lifting is done by a **vendored backtest/factor engine**, carried
*inside* the harness package (`harness/_vendor/`, fully self-contained — the
harness never reaches outside its own tree) and wired in through one seam
(`harness/infra_engine.py`):

- **BacktestEngine** — the backtest engine, reused stage-by-stage. Its
  `Position` turns the cross-sectional signal-strength panel into an
  executable target-weight book (gross budget, warm-up, hold/rebalance, event
  masks, calendar alignment); that book is executed through its **vectorbt**
  portfolio via `PortfolioEngine` (weights run at each bar's OPEN; the harness
  lags them one bar), and NAV + the performance metric map come straight from it.
- **FactorEngine** — the factor-computation engine. Only its **compute + custom**
  path is used: the composed signal is registered into a `FactorManager` as a
  *custom* factor cube so FactorEngine owns the cube / cache / manifest (lineage).
  The predefined indicator library is intentionally **not** exposed.
- **CTX** — data cleaning/loading. Its `CTX` pipeline turns daily
  parquet into the panel the rest of the harness consumes — both user-supplied
  parquet and online yfinance pulls (transcoded to parquet) share this tail.

## What you control

- **Objective is a free choice.** Optimise and gate on any of
  `sortino, sharpe, calmar, cagr, ann_return, hit_rate, maxdd, ann_vol,
  avg_turnover` — each knows whether higher is better — and set the **Pass Line**
  to any value. Metrics come from BacktestEngine: excess returns over a pinned 2%
  risk-free rate, annualised on `annualization` periods per year (252 by default).
- **The degenerate book can't win.** Every book the harness builds is
  cross-sectionally demeaned, so its *net* exposure is ~0 by construction — which
  is why the activity gate is on **gross** exposure, `avg_gross` (mean per-bar
  sum |w|), not net. A book whose average gross exposure is ≤ `1e-8`
  (`objective.ACTIVITY_EPS`) is *inactive*: it can never win the in-sample search
  and can never Pass, so objectives like turnover or vol can't be gamed by
  holding nothing.
- **`gross` is real leverage.** It is applied exactly once, by the engine:
  the harness hands `Position` a **unit-gross** strength panel and `Position`
  scales it to `gross_target`; the same value is passed to the vbt portfolio as
  `max_gross_exposure`, so `gross = 2.0` executes a levered book instead of being
  clamped back to 1.0.
- **Three data sources.** `synthetic` (offline GBM basket); `yfinance` — pulled
  **online**, transcoded to a local parquet, then cleaned through the
  `CTX` pipeline; or `ctx` — your own daily **parquet** loaded through the same
  `CTX` pipeline (set `parquet_daily` to the path(s)). yfinance and ctx share one
  ingest tail, differing only in where the parquet comes from.
- **Eight input fields, and two of them are easy to confuse.** A formula may
  reference `open, high, low, close, volume, typical_price, vwap, returns`.
  `typical_price` is **always** `(high + low + close) / 3` — an unweighted
  intraday average with no volume term in it. `vwap` is a genuine volume-weighted
  price **only** when `data_source = "ctx"` and every selected symbol's parquet
  carries a `Vwap` column (CTX volume-weights it on resample); yfinance and the
  synthetic generator ship no VWAP at all, so there `vwap` falls back to the same
  HLC3 proxy. Which one a panel actually holds is recorded in `Panel.vwap_source`
  (`source` | `hlc3`), hashed into the data version, and printed in the
  provenance string (`synthetic(vwap=hlc3)`), and the D2 alignment check reads it
  so a "volume-weighted execution" story cannot be stamped consistent against a
  panel that has no volume in the field.
- **Tri-alignment gate** (`tri_align`, **on by default**). Machine-verifiable
  consistency between the story and the maths, so a factor cannot ship a
  plausible-but-post-hoc rationalisation:
  - **Mandatory static admission.** The compiler enforces typed parameters,
    dimensional consistency and the precise single-construction grammar above.
    It is independent of `tri_align`. Static rejects become IS dead ends before
    numerical evaluation or scoring and cost no extra model completion.
  - **D2 — mention (warn).** Every field the formula depends on must be named or
    alluded to in the rationale/mechanism.
  - **D3 — sign (warn).** The realised in-sample IC sign must match the declared
    `expected_sign`. The IC is strictly point-in-time: the terminal pair whose
    forward return would land after T_n is dropped.

  D2/D3 are surfaced, recorded on the ledger row and fed back to the proposer —
  never fatal. Both are deterministic and provider-free. Economic causation remains
  a hypothesis; no LLM Judge certifies or selects candidates.
- **Interpretability toggle.** Turn it on and the model is *asked* for a
  rigorous, factor-specific economic `rationale` **and** `mechanism`. The gate is
  structural, not semantic: both fields are mandatory, each must clear a length
  floor, they must be distinct from each other, and neither may be a verbatim
  restatement of the formula — a proposal failing any of these is rejected.
  Genuine economic depth is strongly prompted but not machine-graded (that would
  put an LLM back inside the decision, which the design forbids).
- **Windows two ways.** The "Zora clock" (`research_date` T_n + `is_years` N +
  OOS policy) or explicit
  `is_start/is_end/oos_start/oos_end` (both bounds of a window required, or
  neither). The OOS window is always forced to start after the IS window ends —
  no bar is ever in both. Each ledger row stores the actual scored bounds in
  `is_start/is_end/oos_start/oos_end` and preserves the requested calendar
  bounds separately as `configured_is_*` and `configured_oos_*` provenance.
- **Two OOS clocks.** `oos_mode="fixed_days"` preserves the original
  `oos_days` business-day horizon. `oos_mode="per_step"` evaluates each walk
  step from T_n through the last actual panel bar before the next research date.
  This avoids overlapping QS + 252-day holdouts and does not mistake weekends or
  exchange holidays for bars.
- **Data coverage is a hard preflight.** Before a walk can reset its artifact
  directory or the TUI can replace its log, the runner validates the configured
  range and loaded panel against the source's expected first/last sessions.
  `per_step` calendar cutoffs on weekends or exchange holidays do not require
  nonexistent bars; missing expected sessions still fail. Synthetic uses its
  generator's business-day calendar; real equity calendars use
  `exchange-calendars` (included in `.[data]`). Unknown or mixed calendars fail
  explicitly. The walk loads one panel for all stops. Resume validates its content
  hash against the checkpoint before changing manifest, ledger or trace.
- **IS and OOS floors count live bars.** `min_is_days` and `min_oos_days` are
  floors on `n_active` in their respective windows: bars on which the book
  actually held exposure, rather than calendar bars. The optimiser will not
  promote a candidate below the IS floor, and the verdict enforces both. The
  engine scores a flat bar as a `0.0` return rather than NaN, so a strategy that
  was at risk on three days out of sixty would otherwise clear a 20-day floor on
  57 vacuous zeros.
- **Search breadth × depth.** `max_iters` sets the refine **depth** (propose→backtest
  rounds per research date); `candidates_per_round` sets the **breadth** — how many
  distinct candidate factors the model returns in a *single* call each round, so
  widening the search costs no extra LLM completions and the recorded trace still
  holds exactly one propose completion per round (`replay` stays bit-for-bit). Both
  are config fields and `run` flags (`--max-iters`, `--candidates-per-round`);
  breadth defaults to 1 — byte-identical to the classic one-proposal-per-round.
- **Walk-forward step.** `frequency` is any pandas offset alias that advances at
  least one bar: `YS` (default), **`QS` (quarterly)**, `MS`, `W`, `B`, `D`, a
  multiple (`2QS`, `6MS`, `13W`) or an anchor (`QS-FEB`, `W-MON`). It is
  validated when the config is built, and the walk's schedule is computed
  before anything on disk is touched — a typo used to delete the previous run's
  ledger *and* its LLM trace. Note that pandas 3 **removed** the old `Y`/`Q`/`M`
  spellings: use `YS`/`QS`/`MS` (period start) or `YE`/`QE`/`ME` (period end).
- **Worker processes.** `n_jobs` runs the K candidates of one round through
  separate processes (`--n-jobs`, or **Workers** in the TUI). It parallelises
  **breadth only** — depth reads the previous round's metrics and each date
  reads the earlier dates' verdicts, so both are sequential by construction.
  Proposal completions and ordered state updates stay in the parent. With the default `candidates_per_round = 1` there is one backtest
  per round and workers have nothing to do. The backtest is ~97% of a round's
  wall clock, so the speedup tracks the worker count: measured 4.3–4.7× at K=6 on
  a 10-name, 15-year panel — but only once the pool is *warm*, since each worker
  pays a few seconds of engine import and numba compilation on its first task, so
  a one-round run is no faster. `n_jobs = 1` (the default) takes the historic
  inline path, and above 1 every result is folded back **in candidate order**, so
  the prompts, the ledger, the trace and the winner are identical either way —
  `tests/test_parallel.py` records a walk *with* workers and replays it *without*.
  (Threads were measured and rejected: nothing on the hot path releases the GIL.)
- **Paced retries.** A rate limit, a network blip or a CLI crash on one round is
  transient: that round is skipped and the **next round re-asks the same
  question**, which is why a retried search still records exactly one completion
  per round. `retry_backoff` (default 1.0s) is the pause before the first retried
  round and doubles per *consecutive* failure up to `retry_max_delay` (30.0s);
  `retry_backoff = 0` disables the pause. Because the retry *is* the next round,
  the retry budget is `max_iters`. An **auth-shaped** failure is not treated as
  transient — it is re-raised at once and translated into a setup guide, because
  it would fail identically every round.
- **Six providers.** `scripted` (offline), `claude`, `openai`, `deepseek`,
  `codex` (ChatGPT-login CLI), `opencode` (CLI). The core imports no vendor SDK;
  each adapter loads lazily only when selected.
- **Prompts live in JSON.** All 13 LLM prompts are in `harness/prompts/*.json`,
  editable by hand (see [Editing prompts](#editing-prompts)).

## Design in one breath

- **LLM invents the maths** — operators are just building blocks; no human factor pool.
- **No look-ahead** — time-series operators only read past rows (negative windows
  rejected), and positions are lagged one bar before meeting returns.
- **OOS is the arbiter** — optimise all you like in-sample; the untouched OOS
  window decides whether it counts.
- **Reproducible** — content-hashed data version + run manifest, and every LLM
  completion is recorded so `replay` re-drives a run bit-for-bit without calling
  the model (it reloads the data — offline for `synthetic`, a re-read for
  yfinance/ctx — and warns if the data version drifted). The execution layer is
  fully seeded; live providers request `temperature=0` for best-effort
  determinism where the model accepts it (reasoning models that reject it fall
  back to the API default), but only the recorded trace is truly bit-exact —
  Anthropic's API has no seed, OpenAI's is fingerprint-bounded. The CLI providers
  decode the model's answer as UTF-8 explicitly, so a recorded trace is
  byte-identical across operating systems (under the platform default a Windows
  box decoded `≥`, `—` or `σ` as cp1252 mojibake and poisoned every later prompt).
 - **Structured output is backend-dependent** — direct OpenAI calls transmit
  `response_format=json_schema`; direct DeepSeek calls transmit
  `response_format=json_object` plus `thinking=enabled` and the required JSON
  prompt hint; direct
  Claude calls transmit `output_config.format=json_schema`. OpenCode uses
  its local server's `format: json_schema` StructuredOutput tool when the selected
  model supports tool choice. Its hosted Go fallback is model-routed: most Go
  models use `/v1/chat/completions`, while `gpt-5.6-luna` uses `/v1/responses`.
  Go's OpenAI-compatible wire format does not guarantee native schema support:
  DeepSeek currently rejects `json_schema`, so the adapter uses JSON-object mode
  with thinking enabled; other models may accept schema but expose result text
  differently. OpenAI/Claude wire schemas use closed objects and a parameter-entry
  array (`name`, `type`, `value`); Claude's unsupported array length constraints
  are omitted on the wire. The local parser converts entries back to bindings
  and enforces exact candidate counts, unique names and all mathematical rules.
  The harness still validates every result locally.
  OpenCode 1.18.x has no effective seed control; its temperature is agent-level
  and its CLI output cap is experimental. The legacy `opencode run --format json`
  path still wraps events only and is not schema-constrained.
- **The LLM never touches** backtesting, statistics, or the Pass/Fail decision.

## Known biases and limits

None of these are bugs; they are properties of the study you are running, and
they are here rather than in a footnote because they change what a **Pass**
means.

- **The built-in universes are survivorship-biased.** `config.ASSET_PRESETS` —
  what `symbols: "all"` expands to — is a list of tickers that exist **today**.
  Any historical study over them is conditioned on survival, which flatters every
  metric. If you care about the answer rather than the mechanics, supply a
  point-in-time universe of your own.
- **Incomplete symbol coverage is FATAL by default,** precisely so that
  survivorship cannot creep in silently: if the source returns no usable Close
  data for a requested symbol, the run stops instead of quietly studying the
  remainder. `allow_missing_symbols = true` opts into the survivors-only
  cross-section and emits a `RuntimeWarning` naming the dropped names. It is
  deliberately **not** in `runner._RESUME_IGNORE` — flipping it changes the
  cross-section, so it blocks a `--resume`. (A name that trades for part of the
  window and then delists is *not* missing: it keeps its history and enters the
  panel normally.)
- **yfinance prices are retroactively adjusted.** The harness pulls with
  `auto_adjust=True`, so today's split/dividend factors are applied to the whole
  history and the price at date *t* is not the price a trader saw at *t*. That is
  a real (if mild) look-back in the *input data*, which is why it is stamped into
  the provenance string — `yfinance(auto_adjust=True,vwap=hlc3)` — and hence into
  `data_source`/`data_version` on every run record.
- **The cross-run journal costs independence, not point-in-time discipline.** The
  journal *is* gated (see [The journal feeds back](#the-journal-feeds-back-memory--true-on-by-default)),
  but consulting the same OOS window across many runs makes the reported Pass
  rate optimistic: a Pass Line cleared after twenty journal-informed runs is
  weaker evidence than a cold one.
- **The offline demo is a mechanism test, not a result.** Synthetic GBM prices
  contain no reversal signal to find; the demo exists to prove the loop runs, and
  it ends FAIL.
- **A Pass is one held-out window.** No transaction-cost sensitivity, no capacity
  model, no borrow costs, no regime analysis, no multiple-testing correction
  across the factors a walk discards. Costs are a single `cost_bps` slippage on
  turnover. Outputs are hypothetical backtests, not investment advice, and
  nothing here is a recommendation to trade.

## Quick start (offline — no API key, no network)

"Offline" here means no network and no API key: the `scripted` provider and the
`synthetic` data source need neither. It does **not** mean zero dependencies —
the backtest/factor engine is the vendored engine under `harness/_vendor/`, so
install its runtime deps first (this also installs what the test suite needs):

```bash
cd harness
pip install --require-hashes -r requirements.txt
pip install --no-deps --no-build-isolation -e .
```

The first command materialises the complete reference environment and verifies
every downloaded wheel or sdist against SHA-256 hashes in `requirements.txt`.
The second puts this checkout on the path without resolving any dependency
outside that lock. `pyproject.toml` states a resolver eligibility range
(`>=3.11,<3.15`); validated runtime support is Windows / CPython 3.14.
Linux/macOS engine execution and other Python runtime versions are unverified.

`pip install .` (non-editable) works too and installs the `zora-harness` console
script, the prompts and the vendored engine — but a wheel install has no
`README.md`, `tests/`, `run_config.json` or `data/` next to the package, so the
commands below that reference those paths assume the source checkout.

```bash
python -m harness.cli doctor            # can a run start on this machine?
python -m harness.cli run --no-memory   # single research date -> Pass/Fail
python -m harness.cli show              # recorded factors + verdicts
python -m harness.cli log               # run manifest
python -m harness.cli tui               # interactive TUI (all params exposed)
python tests/run_tests.py --workers 4   # full offline suite, disjoint shards
```

### What the demo actually prints

Verbatim, from `python -m harness.cli run --no-memory` — no `--config`, so this
is the built-in defaults, which the shipped `run_config.json` mirrors field for
field (scripted provider, synthetic data, seed 17):

```
T_n=2023-01-01  universe=10  objective=Sortino  pass_line=1.0  data=synthetic(vwap=hlc3)@5fa07d71
  [iter 0] IS Sortino=-1.3749  -delta(close, 5)
  [iter 1] IS Sortino=-1.6392  -rank(delta(close, 5))
  [iter 2] IS Sortino=-1.2418  -rank(delta(close, 5)) * rank(ts_std(returns, 20))
  [iter 3] IS Sortino=-1.2345  -rank(ts_mean(returns, 20))
  [iter 4] IS Sortino=-2.3943  -rank(delta(close, 5)) * rank(volume / ts_mean(volume, 20))
  [iter 5] IS Sortino=-1.4839  -rank(decay_linear(returns, 10))
  [iter 6] IS Sortino=-1.1550  -rank(delta(close, 5)) / rank(ts_std(returns, 20))
  [iter 7] IS Sortino=-0.9545  -rank(delta(close, 5)) * rank(ts_mean(high - low, 20) / ts_mean(close, 20))
  -> best -rank(delta(close, 5)) * rank(ts_mean(high - low, 20) / ts_mean(close, 20))
  -> IS Sortino=-0.9545  OOS Sortino=1.2061  => FAIL

FAIL: -rank(delta(close, 5)) * rank(ts_mean(high - low, 20) / ts_mean(close, 20)) -> artifacts\runs\demo
```

Eight rounds, eight **distinct** proposals: the scripted provider's script is
exactly `max_iters` long on purpose, because a shorter one made the demo re-print
its last formula five times (the provider holds on its final response), which
reads like a hung search.

### Why the demo ends FAIL

Synthetic prices do not establish a predictive edge. Clearing an objective's
OOS Pass Line alone is also insufficient when `require_sign_consistency=true`:
both IS and OOS must have finite, strictly positive **CAGR**, measured from net
compounded returns. This evidence rule is independent of the chosen objective.
A 60% hit rate can still lose money if losses outweigh wins; positive arithmetic
Sharpe/Sortino can coexist with negative compounded growth under volatility drag.
Setting the flag to `false` disables only this profitability gate; sample floors,
exposure and the OOS Pass Line still apply.

### More things to run

```bash
python -m harness.cli run --walk-forward --config run_config.json
python -m harness.cli run --walk-forward --max-iters 6 --candidates-per-round 4  # depth x breadth
python -m harness.cli run --walk-forward --frequency QS --n-jobs 4               # quarterly, 4 workers
python -m harness.cli replay                                                     # re-derive it, no model call
python -m harness.cli memory                                                     # rebuild the research journal
```

## Command reference

The installed console script is **`zora-harness`**; `python -m harness.cli` is
the exact equivalent from a source checkout. Every subcommand takes
`--config PATH`, which defaults to the **built-in defaults**, *not* to
`run_config.json` — pass it explicitly to use the file. All of them except
`configure` also take the shared overrides `--run-name`, `--provider`,
`--data-source` and `--research-date`, which are applied on top of the config and
then re-validated, so a bad override (`--max-iters 0`) fails cleanly instead of
deep inside a run.

One exception: **`tui`**. A config saved to the default path (`run_config.json`
in the launch directory, via **Save** / `ctrl+s`) is loaded automatically on the
next launch, so an interactive session carries your last-saved state instead of
starting from the built-in defaults every time. An explicit `--config PATH`
still loads that file, and a missing file there still errors.

| command | what it does |
|---|---|
| `configure` | write a validated `run_config.json` from defaults (or `--config`) plus `--set KEY=VALUE` overrides |
| `run` | mine a factor at one research date, or `--walk-forward` across the whole schedule |
| `show` | one row per recorded research date: verdict, IS value, OOS value, formula |
| `log` | the run manifest: provider, model, seed, data version, factor/pass counts |
| `replay` | re-drive a recorded run from its cached LLM trace, with no model call |
| `memory` | rebuild / inspect / extend the persistent research journal |
| `tui` | launch the Textual UI over the same config and runner |
| `doctor` | answer "could a run start right now?" and print the fix if not |

**`configure`** — `zora-harness configure [--config BASE] [--out PATH] --set KEY=VALUE ...`

Each `VALUE` is parsed as **JSON first** and falls back to a bare string, so
`pass_line=1.2` is a float, `require_interpretability=true` is a bool,
`symbols=["AAPL","MSFT"]` is a list, and `objective=sortino` is a string. The
merged config is re-validated through `RunConfig` before anything is written: an
**unknown key or a rejected value exits 2 and writes nothing**. `--out` defaults
to `--config`, and then to `run_config.json`.

```bash
zora-harness configure --set provider=claude --set pass_line=1.0 --out run_config.json
zora-harness configure --config run_config.json --set frequency=QS --set oos_days=60
```

**`run`** — blocked up front by the readiness gate; see
[First-time setup](#first-time-setup-keys--login). `--walk-forward` steps T_n
from `t_0` to `t_p`; `--resume` continues a killed walk from its ledger (skipping
done dates and re-seeding the learning context, refusing if the config changed on
anything that would desync the trace); `--no-memory` keeps the run out of the
durable journal.

**`replay`** — re-drives the recorded walk from
`artifacts/runs/<run_name>/llm_trace.jsonl` with **no provider call**, then diffs
the deterministic slice of every record (`research_date`, `formula`,
`expected_sign`, `is_metrics`, `oos_metrics`, `verdict`, `data_version`) against
what was recorded. It **exits 1** if any date fails to reproduce bit-identically,
if the trace length does not match what the config consumes, if trailing
completions were never replayed, or if any trace entry has no prompt hash. A
prompt-hash mismatch or a drifted data version is a `RuntimeWarning`: the records
may still match, while their provenance is explicitly marked suspect.

**`memory`** — `--list` (one line per archived run), `--archive` (pull the
current run's ledger in), `--synthesize` (spend ONE model call on a narrative).
With no flag it regenerates `ResearchNotes.md` from `raw/`.

**`doctor`** — `--provider NAME` checks one provider instead of the config's;
`--all` prints a status line for every provider plus the shared runtime block.

**Exit codes.** `doctor` exits `0` when a run could start and `1` when it could
not; under `--all` it lists every provider and exits `1` only if the shared
*runtime* is broken, since one provider missing a key does not make the machine
unrunnable. `run` exits `3` when the readiness gate blocks it or
an auth-shaped failure is translated into a setup guide, and `3` on a
`PromptAssetError`. `replay` exits `1` on any divergence. `configure` exits `2`
on a bad key or value. `show`/`log`/`memory --list` exit `1` when there is
nothing recorded to print.

## The TUI

`python -m harness.cli tui` opens a **left rail + main pane** layout rather than
one tall column:

```
 ╔═╗ ╔═╗ ╦═╗ ╔═╗   ▕ Data  Windows  Backtest  Model  Advanced  Guide
 ╔═╝ ║ ║ ╠╦╝ ╠═╣   ▕
 ╚═╝ ╚═╝ ╩╚═ ╩ ╩   ▕  █ hint for this tab
 ▁▁▂▃▂▄▅▄▆▅▇▆█▇▆█▇ ▕
 words → factor →  ▕   Source  [ synthetic ▼ ]   Asset class  [ equity ▼ ]
         verdict   ▕           the price source              a preset universe
 ╭─ MISSION ─────╮ ▕  Symbols  [ all                                     ]
 │ provider  ... │ ▕           'all' = the whole asset class, or a list
 │ goal      ... │ ▕
 ╰──────── run ─╯  ▕
      [ Run ]      ▕
  [Check] [Save]   ▕
 ╭─ LOG ─────────────────────────────────────────────── idle ─╮
```

- the **rail** carries the wordmark, a **DECISION** block (Objective, Pass Line,
  Interpretability), a **MISSION** card (a live digest of the
  provider / model / objective + Pass Line / universe / data source / search
  depth × breadth / memory / mode, titled with the run name) and the action
  buttons, docked so **Run** stays reachable on a short terminal;
- each page is a **4-column grid** — two label/field pairs per row — so a form is
  half as tall as it used to be;
- **every field is captioned** with a one-line gloss directly underneath it. The
  caption sits in the field's own grid cell and takes the row gutter's place, so
  documenting the exposed knobs stays compact; longer Backtest and Advanced
  pages scroll below a 120×40 terminal rather than cropping fields;
- the **Guide** tab renders this README, so the manual is a keystroke away
  (`ctrl+g`) with a clickable table of contents. Opening it slides the rail out
  of the way for reading width; leaving it brings the rail back. (A wheel install
  has no README next to the package, so there the tab says so instead.)
- the **log** spans the full width and its frame doubles as the busy indicator
  (`idle` / `running…`);
- the layout is **responsive**: below 112 columns the forms fall back to one pair
  per row and the rail narrows; below 84 the Guide drops its contents sidebar;
  below 32 rows the sparkline and tagline give way to the MISSION card;
- **one design language**, defined once in [`harness/palette.py`](harness/palette.py):
  black / dark grey / pale gold. Hierarchy is luminance, not hue — bronze frame →
  brass active border → gold title. Only PASS/FAIL-shaped meaning gets its own
  colour (sage / amber / terracotta), and the log tags with those same hexes
  rather than terminal colour names, so `preflight`'s readiness block belongs to
  the same picture.

The TUI exposes the run fields directly. Guardrails and automation that are easy
to misuse live under **Advanced**: data bounds, OOS mode, warmup, missing-symbol
policy, alignment annotations, token/retry budgets and the config path.

**Reasoning effort** is on **Model**, next to Workers. For `provider="codex"`,
it sends `model_reasoning_effort` to every proposal and memory-synthesis
call. `Default?` leaves the client's setting untouched and is explicitly labelled
unverified; choose a named effort when it must be pinned.
The picker reads the installed Codex model catalog, with documented fallback
capabilities when that public cache is unavailable. GPT-6 Luna advertises `low`,
`medium`, `high`, `xhigh` and `max`, not `ultra`. Other providers grey out this control.
Seed lives under Advanced.
ChatGPT OAuth remains the default Codex authentication route.

Codex always receives an explicit `-m` model id. Its provider-specific defaults
are `gpt-6-luna` for proposals; incompatible ids (including old
Claude ids) fail before inference instead of falling back to the client's model.
Switching providers in the TUI replaces only the previous provider's default ids
and logs that change; explicitly customized ids are preserved and validated.
Codex does not support LLM `seed`, `temperature` or `max_tokens` through this
adapter. Temperature and Max tokens are disabled in the TUI; Seed remains active
for local data/engine reproducibility, without promising deterministic inference.
Codex traces record requested/selected model, effort and unsupported controls.
Selection is verified at the CLI invocation boundary, not reported as a
server-side model identity. An inherited client-default effort remains unverified.

**Quit / ctrl+q during a run** requests cancellation and shows `stopping…`.
The app closes after provider subprocesses and backtest workers are cleaned up,
without starting another model call or publishing the interrupted factor as
completed. Windows provider trees are owned with a Job Object; other platforms
use a process group. An inline backtest or SDK request already executing finishes
before the cooperative cancellation checkpoint, so the UI can remain open
briefly while stopping. Existing completed dates remain available for resume.

After moving/renaming a virtual environment's directory, regenerate its installed
entry points with the environment's Python (`python -m pip install --no-index
--no-deps --no-build-isolation --editable .` for this source checkout). The
generated `zora-harness.exe` embeds an absolute interpreter path; moving the
folder alone does not update it.

**The Walk-forward switch** (in **Windows**) picks between the two things the
harness can do, and the two modes read **different** window fields — so the ones
the current mode ignores are greyed out rather than left looking live:

| | Walk-forward **off** | Walk-forward **on** |
|---|---|---|
| what runs | one factor at `research_date` (T_n) | one factor at *every* T_n from `t_0` to `t_p` |
| reads | `research_date`, and `is_start/is_end` + `oos_start/oos_end` if **both** bounds of a window are filled | `t_0`, `t_p`, `frequency` |
| ignores | `t_0`, `t_p`, `frequency` | the explicit overrides — `runner._step_config` clears them at each stop |
| learning | none | each date sees earlier dates' factors whose OOS window has already closed |

`is_years` feeds both modes: IS = `[T_n − is_years, T_n]`. In `fixed_days`, OOS
is the next `oos_days` business days; in `per_step`, OOS ends at the next
research stop and validation counts only actual panel bars.

With `oos_mode="per_step"`, a QS walk normally gives each quarter roughly 63
actual daily bars, so adjacent OOS windows do not overlap and the next quarter
can learn from the previous closed verdict. With `fixed_days=252`, learning stays
point-in-time but the first four quarterly stops have no closed prior window.
And an *anchored* quarter starts on its own anchor, so `QS-DEC` from
`t_0 = 2020-01-01` begins at 2020-03-01 and drops T_0 — plain `QS`
(Jan/Apr/Jul/Oct) is the one that honours a January T_0.

Keys: `ctrl+r` run, `ctrl+d` check-setup, `ctrl+s` save, `ctrl+l` load, `ctrl+g`
guide, `ctrl+q` quit. Numeric fields validate as you type and date fields carry a
`YYYY-MM-DD` placeholder.

- **Symbols** default to **`all`** — every ticker of the selected asset class
  (switch equity→crypto and `all` follows) — or type an explicit comma-separated
  list to narrow it (`symbols: "all"` works in a config file / `--set symbols=all` too).
- **Provider setup, in-app.** Switching the Provider dropdown prints its readiness
  at once; for a key provider (claude/openai/deepseek) paste the key into the
  **API key** box and press **Save key to .env** — it is written to the git-ignored
  `.env` *and* made active immediately (no restart), and the value is never echoed
  or logged. CLI providers (codex/opencode) are pointed at their one-time login.
- **ctx parquet auto-discovery.** The parquet fields activate only for
  `data_source = ctx`; set the **Data dir** (default `data/`) and the parquet box
  auto-fills with the CTX-named parquet found there (or press **Scan**), so paths
  need not be hand-typed. See `harness/data/README.md` for the required filename
  convention.

Runs happen on a worker thread so the UI never freezes. (Needs the `tui` extra:
`pip install '.[tui]'`.)

## Configuration reference

A `RunConfig` is JSON-serialisable, so one file fully describes (and reproduces)
a run. Every field below exists; `RunConfig.from_dict` **rejects an unknown key**
(so a typo cannot sit silently in a config), including removed `simplified` and
Judge fields, and re-validates on every load and every CLI override. Recorded
execution additionally requires the complete current schema.

### Universe & data

| field | default | what it does |
|---|---|---|
| `symbols` | `["all"]` | The universe. `"all"` (string or single-element list) expands to `ASSET_PRESETS[asset_class]`; a bare or comma-separated string is normalised to a list. An empty universe is rejected. |
| `asset_class` | `"equity"` | `equity` / `etf` / `crypto` / `fx`. Selects the `all` basket, and the CTX asset+region token for a yfinance transcode. |
| `data_source` | `"synthetic"` | `synthetic` / `yfinance` / `ctx`. |
| `data_start` | `"2015-01-01"` | First bar loaded (the whole panel, not the IS window). |
| `data_end` | `"2024-12-31"` | Last bar loaded. |
| `parquet_daily` | `null` | Path or list of paths for `data_source="ctx"`. A **relative** path resolves against the harness root, not the CWD, so a config can ship with the repo. |
| `data_dir` | `"data"` | Folder the TUI's **Scan** button searches for CTX-named parquet. A scan hint only — no execution path reads it, so it never blocks a `--resume`. |
| `allow_missing_symbols` | `false` | Opt in to running on the survivors when the source cannot supply every requested symbol. See [Known biases](#known-biases-and-limits). |

### Research clock

| field | default | what it does |
|---|---|---|
| `research_date` | `"2023-01-01"` | T_n — "today" for the researcher. Overridden per stop in a walk. |
| `is_years` | `5` | In-sample window length in calendar years: IS = `[T_n − is_years, T_n]`. |
| `oos_days` | `252` | Out-of-sample horizon after T_n, in **business** days. |
| `oos_mode` | `"fixed_days"` | `fixed_days` keeps the historic horizon; `per_step` runs OOS to the next research stop using actual panel bars. |
| `walk_forward` | `false` | Persisted TUI/CLI mode. The `--walk-forward` CLI flag also enables the walk. |
| `t_0` | `"2020-01-01"` | Walk-forward start. |
| `t_p` | `"2024-01-01"` | Walk-forward end. Exempt from the resume guard so a run can EXTEND its horizon — but only upward: shrinking it would drop already-recorded dates off the schedule while their ledger rows stay on disk, so that is refused. |
| `frequency` | `"YS"` | Walk-forward step; any pandas offset alias daily-or-coarser. Validated at construction. |
| `is_start` / `is_end` | `null` | Explicit IS window. Both or neither. |
| `oos_start` / `oos_end` | `null` | Explicit OOS window. Both or neither. |
| `warmup` | `0` | Extra business-day pre-roll before the first IS date for rolling operators. The required data start is IS start minus this value. |

### Backtest

| field | default | what it does |
|---|---|---|
| `cost_bps` | `3.0` | Per-unit-turnover cost in basis points, mapped to vbt **slippage** (the engine models slippage, not commission). |
| `gross` | `1.0` | Gross exposure budget (sum \|w\|) per bar, applied once by `Position` and passed to the portfolio as `max_gross_exposure`. `> 1.0` is genuine leverage. |
| `annualization` | `252` | Periods per year pinned into the metric annualisation, so IS/OOS slices are scored on the trading-day clock rather than the engine's 365-calendar-day default. |
| `initial_cash` | `1000000.0` | NAV base for the vbt portfolio. |
| `selection_mode` | `"none"` | `none` keeps all non-zero signal weights; `top_q` keeps signed quantiles; `top_k` keeps a fixed K on each signed side. |
| `top_q` | `0.20` | Fraction selected on each signed side when `selection_mode="top_q"`. |
| `top_k` | `10` | Number selected on each signed side when `selection_mode="top_k"`; the TUI disables this field for other modes. |
| `hold_every` | `1` | Number of bars an entry remains in the rolling held book. |
| `rebalance_every` | `1` | Recompute entries every N bars; intervening bars do not open new positions. |
| `close_delisted_at_last` | `false` | On the first observed missing Open, causally cash-settle any carried position at the most recent prior Close. It never scans future rows or rewrites a past target; a temporarily missing name may reopen later. |

### Objective & pass gate

| field | default | what it does |
|---|---|---|
| `objective` | `"sortino"` | Any key in `objective.METRICS`. Optimised in-sample and gated out-of-sample. |
| `pass_line` | `1.0` | OOS threshold. May be negative (e.g. a `maxdd` line of `-0.2`); must be finite. |
| `require_sign_consistency` | `true` | Require finite, strictly positive net CAGR in **both** IS and OOS, independently of the optimisation/Pass-Line objective. Win rate and arithmetic-return ratios alone do not prove compounded profit. |
| `min_is_days` | `20` | Floor on IS bars **with live exposure** (`n_active`). Candidates below it cannot win optimisation or PASS. |
| `min_oos_days` | `20` | Floor on OOS bars **with live exposure** (`n_active`), not calendar bars. |

### Interpretability & alignment

| field | default | what it does |
|---|---|---|
| `require_interpretability` | `false` | Demand a structured economic `rationale` + `mechanism` and enforce the structural floor on them. |
| `tri_align` | `true` | Optional field-mention and causal IC-sign annotations. Mandatory compiler admission is independent of this switch. |

### Model & search

| field | default | what it does |
|---|---|---|
| `provider` | `"scripted"` | `scripted` / `claude` / `openai` / `deepseek` / `codex` / `opencode`. |
| `model` | provider default | Scripted/Claude: `claude-opus-4-8`; Codex: `gpt-6-luna`; OpenAI: `gpt-5.6-luna`; DeepSeek: `deepseek-v4-flash`; OpenCode: `opencode-go/gpt-5.6-luna`. Inherited defaults follow provider switches; explicit overrides are preserved. OpenCode requires `provider/model`. |
| `reasoning_effort` | `null` | Codex-only model-specific effort override. `null` preserves the client default; recorded for replay/resume. |
| `max_iters` | `8` | Refine rounds per research date (search **depth**). Also the retry budget. |
| `candidates_per_round` | `1` | Candidate factors returned per round (search **breadth**), still one provider call. |
| `max_tokens` | `10000` | Output budget where supported; unsupported by the Codex adapter and disabled in its TUI. |
| `temperature` | `0.0` | Sampling temperature where the provider supports it; `0` is best-effort deterministic. |
| `n_jobs` | `1` | Worker **processes** scoring one round's candidates. `1` takes the historic inline path. Wall-clock only, so exempt from the resume guard and run identity. |
| `retry_backoff` | `1.0` | Seconds before the first retried round; doubles per consecutive failure. `0` disables. Wall-clock only. |
| `retry_max_delay` | `30.0` | Ceiling on that exponential growth; `0` disables retry sleeping. Wall-clock only. |

### Memory & misc

| field | default | what it does |
|---|---|---|
| `memory` | `true` | Read/extend the research journal. Lessons are keyed by checked expression **including bindings** and scoring context (objective, universe, costs, clock, data identity and execution policy). Feedback and notes retain parameters and the originating objective label. |
| `seed` | `17` | Seeds local data/engine execution; passed to providers that support an inference seed. Codex has no effective LLM seed control. |
| `logging` | `true` | Cosmetic; exempt from the resume guard and run identity. |
| `run_name` | `"demo"` | The `artifacts/runs/<run_name>/` subdirectory. Validated as a single path segment — no `/`, `\`, `:`, `..`, drive letters or absolute paths. |

## Real data / real models

- Real data: set `"data_source": "yfinance"` (`pip install '.[data]'`). The
  `data` extra includes the CTX/backtest runtime as well as yfinance and pyarrow,
  so this one install supports the actual load-to-execution path.
- Your parquet: set `"data_source": "ctx"` and `"parquet_daily"` to the daily
  parquet path(s) (CTX long format: `Datetime, Symbol, Open, High, Low, Close`
  [`, Volume`][`, Vwap`]); CTX resamples/cleans to the daily clock. A **relative**
  path is resolved against the harness root (like `data_dir`), not the process CWD
  — so a config can ship with the repo instead of pinning one machine's absolute
  path. For a run's *identity* the path is further reduced to a harness-root-relative
  POSIX form, so the same committed config is **one** experiment on Windows and on
  Linux: re-spelling the same file absolute↔relative (or opening it on another OS)
  neither blocks `--resume` nor forks a second journal entry. A parquet stored
  *outside* the harness root cannot be named portably at all and keeps its
  absolute, machine-specific identity. A *different* file still differs, as it
  should.
- Your CSVs: `python data/field_ingest.py` transcodes them into CTX-profiled
  parquet (see [`data/README.md`](data/README.md)).
- OpenAI:    `"provider": "openai"`,  model e.g. `gpt-5.6`,  `OPENAI_API_KEY`.
- DeepSeek:  `"provider": "deepseek"`, model e.g. `deepseek-v4-flash`, `DEEPSEEK_API_KEY`.
- Claude:    `"provider": "claude"`,  `anthropic` + `ANTHROPIC_API_KEY`.
- Codex:     `"provider": "codex"` — run `codex login` once (ChatGPT sign-in),
  needs `codex` on PATH. Answers come from `codex exec --sandbox read-only`.
 - OpenCode:  `"provider": "opencode"`, model `provider/model` e.g.
  `opencode-go/glm-5.2`, needs `opencode` on PATH. Structured requests use a
  local `opencode serve` session and `format: json_schema`; hosted Go fallback
  routes Chat Completions models to `/v1/chat/completions` and
  `gpt-5.6-luna` to `/v1/responses`. DeepSeek Go models fall back to JSON-object
  mode with thinking enabled. Native schema support is model-dependent; the
  harness validates returned JSON locally. The legacy CLI path is pinned to the
  tool-less `harness-llm` agent, and `--auto` is never passed.

Both CLI providers are spawned with an **allowlist** environment — the non-secret
operational variables in `providers/cli_base.py::_ENV_ALLOWLIST` plus at most the
one credential that backend actually needs — so a third-party binary never
inherits the rest of your keyring. See [SECURITY.md](SECURITY.md) for what that
containment does and does not buy.

### Where API keys live

Keys can be exported in your shell *or* stored in this installation's `.env`; a
real shell variable always wins. **Which** `.env` depends on how the harness was
installed:

- **source checkout / `pip install -e .`** (the documented workflow) —
  `harness/_vendor/ENV_MGMT/.env`, git-ignored, documented by the committed
  `.env.template` beside it. Copy the template and fill in `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`.
- **plain wheel install** — a per-user config file:
  `%APPDATA%\zora-harness\.env` on Windows, `$XDG_CONFIG_HOME/zora-harness/.env`
  (falling back to `~/.config`) elsewhere. The package then lives in
  `site-packages`, where writing a credential is wrong twice over: `pip install
  -U` silently deletes it, and a system-wide Python raises `PermissionError`.

`doctor` prints the exact path it will read. A key saved *through the harness*
(the TUI's **Save key to .env**, i.e. `env.save_key`) writes that file with mode
`0600` — owner read/write only, set at creation rather than chmod-ed afterwards,
so it is never briefly world-readable; Windows has no mode bits, so there it
relies on the owner-scoped ACL of its parent directory. If you create the file by
hand from the template, its permissions are whatever your umask gives it. Key
**values** are never echoed, logged or written anywhere else — see
[SECURITY.md](SECURITY.md).

### First-time setup (keys / login)

The `scripted` provider + `synthetic` data need nothing, so the harness runs out
of the box. The first time you pick a **real** provider you almost certainly
haven't set a key or signed in a CLI yet, so instead of dying deep inside the
provider the harness **checks readiness first and prints a setup guide**:

```bash
python -m harness.cli doctor                 # check the configured provider
python -m harness.cli doctor --provider openai
python -m harness.cli doctor --all           # one status line per provider
```

`doctor` is not only about credentials — it answers *could a run start right
now?*, in two halves:

- **the provider** — is the SDK importable, is the API key *set* (never its
  value), is the CLI binary on `PATH`;
- **the local runtime** — are the vendored engine packages present, do they
  actually import (TA-Lib / numba / duckdb / vectorbt are probed for real, which
  costs a few seconds and is the point), and do all 13 prompt assets load.

A ready machine looks like this:

```
OK  provider 'scripted' + local runtime: ready
    ok  offline (no credential needed)
    ok  vendored engine packages
    ok  engine runtime dependencies
    ok  prompt assets
  note: scripted provider: offline canned responses, no key or login.
  note: backtest engine imports and all 13 prompt assets load.
```

The same gate runs automatically before `run` and before a TUI **Run** (the TUI
also has a **Check setup** button); a blocked run prints the guide and exits `3`
rather than starting something doomed. A CLI's *login state* cannot be probed
cheaply or honestly, so it is surfaced as a reminder — and if a run then fails
with an auth-shaped error (a lapsed `codex login`, an expired key), that failure
is translated into the login/key fix instead of a raw traceback.

## The research journal (`memory/`)

`artifacts/` is a **working** directory: it is git-ignored, wiped whenever the
same `run_name` is re-used, and safe to delete. So a finished run is also copied
into `memory/`, which is the durable local record. Its generated JSON and
markdown are git-ignored because they may contain private research hypotheses,
model prose and data provenance. Export or commit selected records only through
an explicit publication review:

```
memory/
  raw/<run_id>.json    one self-contained snapshot per run (manifest + every factor row)
  ResearchNotes.md     the consolidated digest of all of them, regenerated on every write
```

This happens automatically at the end of every `run` (pass `--no-memory` for a
throwaway). A `run_id` is **content-addressed** — `<run_name>-<hash>` over the
config plus the data version — so re-running or `--resume`-ing the *same*
experiment updates one entry instead of littering the journal with
near-duplicates, while changing any real knob starts a new entry. Because an
entry carries the full ledger, the journal still reads correctly after
`artifacts/` is gone.

`ResearchNotes.md` is fully regenerated from `raw/` every time, so it can never
drift from the records: what worked, what didn't (with failure *modes* grouped
by blanking the numbers out of the verdict reasons), which DSL primitives keep
showing up in winners vs losers, and a per-run log. Hand-written commentary goes
in an entry's `"note"` field in `raw/<run_id>.json`, which survives every rebuild
— never in the markdown.

```bash
python -m harness.cli memory                # regenerate ResearchNotes.md from raw/
python -m harness.cli memory --list         # one line per archived run
python -m harness.cli memory --archive      # pull the current run_name's ledger in
python -m harness.cli memory --synthesize   # ONE model call -> a narrative "Synthesis" section
```

`--synthesize` is the only part that costs a completion: it hands the
deterministic digest to the configured model and asks for the *so what* (where
the evidence points / dead ends / open questions). The result is cached against
a content fingerprint of the journal, so once new results land the section is
labelled **STALE** rather than quietly presenting conclusions the record no
longer supports. (The `scripted` provider is refused here: it would write canned
factor JSON into the notes as if it were a narrative.)

### The journal feeds back (`memory = true`, on by default)

The journal is not just a log — it is read back into the proposer prompt. Before
the first research date a run snapshots the accumulated lessons (deduplicated
**by formula**, so a mechanism that died four times is one lesson worth four
times as much, capped at 10 wins / 15 losses) and shows them at every date:

```
RESEARCH JOURNAL --- what you have already learned across 3 previous run(s)...

These FAILED in an earlier run --- treat the mechanism as suspect:
  -rank(delta(close, 5)) x4  (OOS Sortino=-1.2400; OOS Sortino # fails Pass Line (>= #))
```

**This is point-in-time gated.** A journal lesson is an OOS grade, and a grade
computable only from bars after T_n is look-ahead no matter which run produced
it — otherwise the second run of an experiment would be handed, at simulated
T_n = 2023-01-01, the score a formula earned on the holdout window that *starts*
there. So a lesson is withheld until its own OOS window has closed by this date's
T_n, on the **originating** run's `oos_days` (a different experiment may have
held out 60 business days where this one holds out 300), and a lesson whose close
date cannot be recovered is withheld: fail safe. It is exactly the rule an
earlier date's OOS score obeys inside a single walk (`runner._prior_oos_closed`).

What the gate cannot remove is the cost in **independence**: consulting the same
OOS window across many runs makes the reported Pass rate optimistic (selection
bias), so a Pass Line cleared after twenty journal-informed runs is weaker
evidence than a cold one.

Three invariants keep this from eating the crown jewel:

- the snapshot a run was shown is **recorded in its manifest** *unfiltered*, and
  the gate is applied at **render** time from the date's own T_n — which is
  deterministic, so `replay` regenerates byte-identical prompts (no prompt-drift
  warning);
- the snapshot is taken **once**, before the first date, so every date in a walk
  sees the same journal — and `--resume` reuses the *recorded* snapshot, not
  today's, so the second half of a walk is prompted like the first;
- with `memory = false` (or `--no-memory`) the block is omitted entirely and the
  prompt is byte-identical to the pre-feature one, so older traces still replay.

The ledger records `n_journal_wins` / `n_journal_losses` per date — counting the
**gated** view, i.e. what the proposer was actually shown — so you can always tell
how informed a given decision was.

## What a run leaves on disk

Two roots, deliberately different.

**The run ledger** goes to `artifacts/`, resolved against the **current working
directory** (`cli._ARTIFACTS`), so running from the harness root gives
`harness/artifacts/`:

```
artifacts/runs/<run_name>/
  factors.jsonl    the ledger: one JSON record per research date
  manifest.json    config + provider/model/seed + data_version + journal snapshot + usage
  llm_trace.jsonl  every LLM completion, appended live -- what `replay` re-drives
```

**Generated state** — the yfinance parquet cache and CTX's own logs — goes
wherever `data._state_dir()` points: the in-repo `artifacts/` tree for a source
or editable checkout (so shipped relative paths and the `.gitignore` keep
working), else the per-user cache location, `%LOCALAPPDATA%\zora-harness` on
Windows or `$XDG_CACHE_HOME`/`~/.cache/zora-harness` elsewhere. That is why a run
no longer drops an `ArchiveDeck/ComputerLog/` tree into whatever directory you
happened to launch from; CTX's logs land under `<state>/ctx/` and the newest 50
are kept.

yfinance's timezone/cookie SQLite cache lives under `<state>/yfinance/sqlite/`,
separate from other applications' yfinance caches. ZORA serializes ticker
downloads (`threads=False`) and coordinates independent runs with an OS lock,
including parquet writes. Request timeout is explicitly 15 seconds (upstream
cookie/timezone calls and retries can make the total longer). Download failures
retain upstream error details; a connection timeout needs network/proxy/Yahoo
availability diagnosis and cannot be fixed by clearing the SQLite cache.

Two things about a ledger record are worth knowing before you parse one:

- **`n` vs `n_active`.** `n` is bars in the window; `n_active` is bars on which
  the book actually held exposure (per-bar sum |w| > `ACTIVITY_EPS`). The engine
  scores a flat bar as `0.0`, so `n` alone cannot tell a fully-invested window
  from one that sat out most of it. Both `min_is_days` and `min_oos_days` are
  checked against the corresponding window's `n_active`.
- **Ratios can be `Infinity`.** `sortino`, `calmar` and `sharpe` are reported as
  `+inf` for an **active** window that made money with no losing bar / no
  drawdown / no variance — the denominator vanished, not the book. That is the
  correct limit, and it wins comparisons and clears any finite Pass Line, as it
  should. An inactive or degenerate book keeps `NaN` and still scores worst. The
  ledger is written with Python's `json`, so those land in the file as the
  non-standard literals `Infinity` / `-Infinity` / `NaN`: `json.load` reads them
  back, a strict JSON parser will not.
- **`usage` may be `None`,** which means *not costed* — the provider reported no
  token counts — and never *cost nothing*. Each ledger row carries the tokens its
  own date consumed (proposal completions) and `manifest["usage"]` carries the
  run-level total, summed off the ledger so a resumed run also costs the dates it
  inherited.

## Editing prompts

Each file in `harness/prompts/` is one prompt with a `template` (list of lines)
using `$name` placeholders (so the JSON braces of the schema need no escaping).
All 11:

```
system_propose.json          the proposer's system prompt (fields, operators, objective)
user_initial.json            first turn: invent a factor at T_n
user_refine.json             iteration turn: the full IS metric set + the objective it is improving
interpretability_block.json  strict business-justification block (injected when require_interpretability)
multi_candidate_block.json   asks for a JSON ARRAY of N candidates (injected when breadth > 1)
prior_factors_block.json     earlier dates' PASS factors, in this walk (learning by doing)
negative_factors_block.json  earlier dates' OOS FAILURES, in this walk (negative memory)
dead_ends_block.json         candidates already rejected at THIS date, with the reason
journal_block.json           the cross-RUN lessons from memory/ (point-in-time gated)
system_synthesis.json        the journal-synthesis system prompt (the "so what")
user_synthesis.json          the deterministic digest + the objective it was scored on
```

Edit them freely; `promptlib.render()` substitutes the runtime values. Two rules
if you add or rename one:

- a new prompt must also be listed in `promptlib.REQUIRED_PROMPTS` (and in
  `pyproject.toml`'s `package-data`, which is enumerated, not globbed);
  `tests/test_prompts.py` pins that tuple to the directory, so a drift fails the
  suite rather than a user's run.
- a missing, renamed or malformed file is caught **before** a run by the
  readiness gate, as a `PromptAssetError` naming the exact path searched and the
  prompts that *were* found. Without that, a typo'd filename used to surface one
  iteration later as a nameless "provider error", get retried every round, and end
  the run with a misleading "no valid active factor proposal" — sending you to
  debug your LLM instead of your typo.

## Layout

```
harness/
  pyproject.toml   packaging: deps (ranges), the zora-harness console script, package-data
  requirements.txt THE LOCK: exact versions plus SHA-256 hashes for the tested environment
  MANIFEST.in      sdist manifest (prunes artifacts/ArchiveDeck/memory, excludes .env)
  run_config.json  serialised RunConfig: the scripted/synthetic demo default
  field_config.json  serialised RunConfig: the real-data (ctx) experiment
  README.md        this file (also rendered by the TUI's Guide tab)
  SECURITY.md      threat model, trusted inputs, credential handling, reporting
  harness/
    config.py      RunConfig: every knob + IS/OOS window maths + validation
    objective.py   metric registry (direction / neutral baseline / activity gate)
    data.py        Panel + synthetic / yfinance / ctx (parquet via CTX) + state dirs
    factor/        operators.py  contract.py  checked_evaluate.py  evaluate.py  errors.py  protocol.py
    backtest.py    signal -> unit-gross strength -> Position -> lagged book -> vbt
    validate.py    IS/OOS split (no overlap) + Pass/Fail verdict
    alignment.py   optional D2 mention / D3 sign annotations; compiler owns admission
    infra_engine.py seam to the vendored engine + BacktestEngine metric adapter
    _infra.py      lazy bootstrap: puts _vendor/ on sys.path
    promptlib.py   loads/renders/verifies harness/prompts/*.json
    prompts/       the 11 LLM prompts, as editable JSON
    providers/     base, scripted, replay (record/replay), cli_base (env allowlist +
                   secret redaction), claude, openai_api (openai+deepseek), codex, opencode
    proposer.py    bounded factor proposer (strict JSON + typed math) + journal PIT gate
    runner.py      optimise loop, walk-forward, resume guard, usage metering, replay
    parallel.py    optional worker processes for a round's candidate backtests
    store.py       JSONL ledger + LLM trace + reproducibility manifest
    memory.py      durable research journal (memory/raw + ResearchNotes.md)
    preflight.py   readiness gate: provider credential + engine + prompt assets
    env.py         the .env this installation reads/writes (checkout vs wheel)
    cli.py         configure / run / show / log / replay / memory / tui / doctor
    palette.py     the one colour definition (black / dark grey / pale gold)
    tui.py         interactive Textual TUI (captioned forms + a Guide tab)
    _vendor/       first-party engine/support code (owner-attested provenance),
                   with local execution and bootstrap patches:
      BacktestEngine/  Calendars Manager Position Signal Utils
                       ForLoopEngine PortfolioEngine VBTEngine
      CTX/             CTX Config DataOps MarketData + CTX.json
      FactorEngine/    Library Manager Params Proxy Specs + FactorEngine.json
      ENV_MGMT/        imports.py (the shared imports bundle) + .env.template
memory/          the durable local research journal (generated records ignored)
  raw/           one JSON snapshot per run, keyed by content-addressed run_id
  ResearchNotes.md
data/            drop-in folder for data_source = ctx (see data/README.md)
  field_ingest.py  CSV -> CTX-profiled parquet transcoder for your own daily files
tests/           pytest suite (install the dev extra / reference requirements):
  run_tests.py     canonical current-only suite; --workers 8 shards
                   files without overlap. Selectors are single-process checks.
  conftest.py      sole current contract; unsupported markers rejected
  regression_support.py
                   unique node-ID accounting, current-only checks, JSON receipts
                   and source fingerprint. pytest-asyncio runs all TUI tests.
                   activity alignment config configure cross_provider data env
                   current_only frequency hardening negative_memory learning lookahead
                   memory metrics multi_candidate objective objective_run oos_floor
                   no_signal parallel pass preflight prompts provider_failure
                   provider_security providers replay replay_trace resume review2
                   store tui vendor walk_forward windows config_truth runtime_fixes
                   strict_math parallel_bootstrap regression_suite
```

A note on the vendored tree, because two things about it used to be documented
wrongly. `_infra.ensure_infra()` now does exactly one thing: it inserts
`_vendor/` at the front of `sys.path`. That single entry resolves all three
engine packages *and* the shared `ENV_MGMT` bundle, and it is inherited by
spawned worker processes (an injected `sys.modules` entry would not be); there is
no namespace shim and no dependency-alias patching any more. And nothing in the
package reads a `.env` at import time — credentials are loaded only when a real
provider is actually built.

The harness runs straight from source (`python -m harness.cli ...`) once the
runtime dependencies are installed, and it also packages: `pyproject.toml` finds
the `_vendor` packages and enumerates every non-`.py` asset that must travel
(the 11 prompts, `CTX.json`, `FactorEngine.json`, `.env.template`), so a built
wheel carries the engine and the prompts rather than an empty shell.
