"""Run configuration --- every knob for one harness run, in one place.

A RunConfig is JSON-serialisable so a run is fully described (and reproducible)
by a single file. Dates are plain ISO strings; window maths lives here so the
rest of the code never re-derives it.

Windows can be given two ways:
  * the "Zora clock": research_date (T_n) + is_years (N) + oos_days, or
  * explicit overrides: is_start / is_end / oos_start / oos_end.
If a window's explicit bounds are set they win; otherwise the clock is used.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

import pandas as pd

from . import objective as _obj
from .factor.protocol import STRICT, require_contract

# Convenience presets the TUI can offer per asset class.
ASSET_PRESETS: dict[str, list[str]] = {
    "equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
               "NVDA", "JPM", "XOM", "JNJ", "PG"],
    "etf": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "HYG", "XLF", "XLK"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
               "ADA-USD", "DOGE-USD", "AVAX-USD"],
    "fx": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
           "USDCAD=X", "USDCHF=X"],
}


# Walk-forward steps the harness ADVERTISES (error text, CLI help, TUI). Any
# pandas offset alias daily-or-coarser is accepted --- see _check_frequency ---
# except legacy aliases removed by pandas 3; these are the period-START steps a
# research clock actually wants.
FREQUENCIES: tuple[str, ...] = ("YS", "QS", "MS", "W", "B", "D")


def _check_frequency(freq) -> None:
    """Reject a walk-forward step pandas cannot build a schedule from.

    ``frequency`` was the one field with no validation, and it is the most
    expensive one to get wrong: :func:`~harness.runner.run_walk_forward` used to
    clear the run directory before the schedule was built, so a typo --- or the
    pandas-2 spellings ``'Y'``/``'Q'``/``'M'``, which pandas 3 REMOVED rather
    than deprecated --- deleted a finished run's ledger and its irreplaceable
    LLM trace, then died in a raw pandas traceback. It is caught here, at
    construction, where nothing has been touched yet.

    Sub-daily steps are rejected on top of what pandas accepts. The panel is
    daily, so two stops inside one bar produce two records carrying the SAME
    ``research_date`` --- which is the ledger's key: ``--resume`` treats the
    second as already done and drops it, and replay then fails on the completion
    count. There is no schedule finer than a bar worth having.
    """
    if not isinstance(freq, str) or not freq.strip():
        raise ValueError(
            f"frequency must be a non-empty pandas offset alias, got {freq!r}; "
            f"try one of {', '.join(FREQUENCIES)}"
        )
    raw = freq.strip()
    legacy = raw.upper().lstrip("0123456789")
    if legacy in {"Q", "M", "Y", "A", "AS", "BQ", "QQ"} \
            or raw == "qs":
        raise ValueError(
            f"unknown walk-forward frequency {freq!r}: legacy pandas alias; "
            f"use one of {', '.join(FREQUENCIES)} (pandas 3 removed the old "
            "Y / Q / M / A spellings)"
        )
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except Exception as exc:      # noqa: BLE001 - pandas raises ValueError/TypeError
        raise ValueError(
            f"unknown walk-forward frequency {freq!r} ({exc}). Give a pandas "
            f"offset alias: {', '.join(FREQUENCIES)} (year / quarter / month "
            "start, weekly, business-daily, daily), a multiple such as '2QS', "
            "'6MS' or '13W', or an anchored one such as 'QS-FEB' / 'W-MON'. "
            "Note pandas 3 REMOVED the old 'Y' / 'Q' / 'M' / 'A' spellings --- "
            "use 'YS' / 'QS' / 'MS' for the period START (what a research clock "
            "wants) or 'YE' / 'QE' / 'ME' for the period END."
        ) from None
    # A step that does not move the clock forward is not a schedule: '0D' makes
    # pandas raise from inside date_range and '-1D' silently yields an EMPTY
    # range. Asking the offset to advance one timestamp settles both, and works
    # for anchored offsets ('QS-FEB', 'W-MON') that a sign check would not.
    probe = pd.Timestamp("2020-01-01")
    if probe + offset <= probe:
        raise ValueError(
            f"walk-forward frequency {freq!r} does not step FORWARD "
            f"({probe.date()} + {freq} = {(probe + offset).date()}), so the walk "
            "has no schedule. Use a positive step, e.g. "
            f"{' / '.join(FREQUENCIES)}."
        )
    try:
        span = pd.Timedelta(offset)          # only sub-daily Ticks convert
    except (ValueError, TypeError):
        # Day and coarser (pandas 3 makes Day a calendar offset, not a Tick):
        # never finer than a bar, nothing left to check.
        return
    if span < pd.Timedelta(days=1):
        raise ValueError(
            f"walk-forward frequency {freq!r} steps {span}, finer than the daily "
            "bar the panel is built on: two stops inside one bar would share a "
            "research_date (the ledger's key), so the second is silently dropped "
            f"on --resume and replay then fails. Use 'D' or coarser "
            f"({', '.join(FREQUENCIES)})."
        )


def _is_all_sentinel(symbols) -> bool:
    """True if ``symbols`` is the "all" sentinel: the string ``"all"`` or a single
    ``"all"`` token (case-insensitive). It means "every symbol of the selected
    asset class" and is expanded to that class's basket in ``__post_init__``.
    """
    if isinstance(symbols, str):
        return symbols.strip().lower() == "all"
    if isinstance(symbols, (list, tuple)) and len(symbols) == 1:
        return str(symbols[0]).strip().lower() == "all"
    return False


@dataclass
class RunConfig:
    # --- universe & data ---------------------------------------------------
    # "all" (the default) = every symbol of the selected asset_class; it is
    # expanded to ASSET_PRESETS[asset_class] in __post_init__ so the rest of the
    # code always sees a concrete list. Give an explicit list to narrow it.
    symbols: list[str] = field(default_factory=lambda: ["all"])
    asset_class: str = "equity"              # equity | etf | crypto | fx (informational)
    data_source: str = "synthetic"           # "synthetic" | "yfinance" | "ctx"
    data_start: str = "2015-01-01"
    data_end: str = "2024-12-31"
    parquet_daily: list[str] | str | None = None   # daily parquet path(s) for data_source="ctx"
    data_dir: str = "data"                          # folder the TUI scans for CTX-named parquet (data_source="ctx")
    # Opt in to running on the SURVIVORS when the source cannot supply every
    # requested symbol. Fatal by default: silently dropping the names with no
    # data is exactly how survivorship bias gets in, and a delisted stock is the
    # one most likely to be missing. Deliberately NOT in runner._RESUME_IGNORE ---
    # flipping it changes the cross-section, so a resume must refuse it.
    allow_missing_symbols: bool = False

    # --- research clock (spiritual Zora params) ----------------------------
    research_date: str = "2023-01-01"         # T_n --- "today" for the researcher
    is_years: int = 5                         # N --- in-sample window length
    oos_days: int = 252                       # out-of-sample horizon after T_n
    # ``fixed_days`` preserves historic business-day windows. ``per_step`` is
    # resolved by the walk to the next research date and uses actual panel bars.
    oos_mode: str = "fixed_days"
    # TUI/CLI session mode. The CLI --walk-forward flag still takes precedence;
    # persisting this field lets the TUI reopen in the same mode.
    walk_forward: bool = False
    t_0: str = "2020-01-01"                   # T_0 --- evolution start (walk-forward)
    t_p: str = "2024-01-01"                   # T_p --- evolution end
    # Walk-forward step: any pandas offset alias daily-or-coarser. YS/QS/MS/W/B/D
    # are the period-START steps a research clock wants; multiples ('2QS', '6MS',
    # '13W') and anchors ('QS-FEB', 'W-MON') work too. Validated in __post_init__
    # (see _check_frequency) --- a typo here used to cost a finished run's trace.
    frequency: str = "YS"

    # --- explicit window overrides (optional; single-run mode) -------------
    is_start: str | None = None
    is_end: str | None = None
    oos_start: str | None = None
    oos_end: str | None = None

    # --- backtest (BacktestEngine vbt engine) ------------------------------------
    cost_bps: float = 3.0                     # per-unit-turnover cost, basis points -> vbt slippage
    gross: float = 1.0                        # gross exposure per date (sum|w|); this IS the cap
    annualization: int = 252                  # periods/year pinned into BacktestEngine's metric annualisation
    initial_cash: float = 1_000_000.0         # NAV base for the vbt portfolio
    # Cross-sectional portfolio construction. ``none`` preserves the historic
    # all-signal book; ``top_q`` selects signed signal quantiles and ``top_k``
    # selects a fixed number on each signed side.
    selection_mode: str = "none"
    top_q: float = 0.20
    top_k: int = 10
    hold_every: int = 1
    rebalance_every: int = 1
    close_delisted_at_last: bool = False
    # Extra pre-roll loaded before the first IS date for rolling factor operators.
    # Zero preserves historic configs; production configs can opt into a safer
    # one-year pre-roll from the Advanced page.
    warmup: int = 0

    # --- objective & pass gate (free choice) -------------------------------
    objective: str = "sortino"                # any key in objective.METRICS
    pass_line: float = 1.0                    # OOS threshold for the objective
    require_sign_consistency: bool = True     # IS & OOS must both be "good"
    min_is_days: int = 20
    min_oos_days: int = 20

    # --- interpretability --------------------------------------------------
    require_interpretability: bool = False    # gate on a structured economic rationale

    # --- tri-alignment gate (hypothesis <-> formula(AST) <-> data) ----------
    # Machine-verifiable consistency between the story and the maths, so a factor
    # cannot ship a plausible-but-post-hoc rationalisation. D1 (single-monomial
    # AST structure) is always enforced by the compiler; D2 (every
    # formula field is named in the story) and D3 (realised in-sample IC sign
    # matches expected_sign) only WARN --- surfaced and fed back to the proposer,
    # never fatal. All three are deterministic and provider-call-free, so
    # record/replay is unaffected. On by default; economic meaning remains an
    # unverified hypothesis. This switch controls annotations, never admission.
    tri_align: bool = True
    math_contract: str = STRICT

    # --- iteration / model -------------------------------------------------
    provider: str = "scripted"                # scripted|claude|openai|deepseek|codex|opencode
    model: str = ""                          # resolved to an explicit provider default
    max_iters: int = 8                        # refine rounds (search DEPTH)
    # candidate factors the proposer returns per round (search BREADTH). Still ONE
    # provider call per round -- the model returns up to this many candidates as a
    # JSON array -- so the recorded/replayed trace holds exactly one propose
    # completion per round regardless of this value. Each candidate is scored on IS
    # and the best across all rounds x candidates is kept. Default 1 preserves the
    # historic single-proposal-per-round behaviour (and byte-identical traces).
    candidates_per_round: int = 1

    # --- execution: worker processes ---------------------------------------
    # How many candidate backtests of ONE round run at the same time, in separate
    # PROCESSES. This parallelises BREADTH and nothing else, because breadth is
    # the only axis in this harness that is causally independent: refine rounds
    # read the previous round's metrics, and each walk-forward date reads the
    # earlier dates' verdicts, so depth and the walk are sequential BY
    # CONSTRUCTION and always will be. With the default candidates_per_round=1
    # there is exactly one backtest per round and n_jobs has nothing to do ---
    # raise K first, then raise this.
    #
    # Threads were measured and rejected: nothing on the hot path (vectorbt's
    # numba kernels, the vendored engine, the pandas reshapes) releases the GIL
    # or runs a parallel kernel, so a thread pool buys ~1x. Processes buy about
    # (workers)x on the backtest stage, which is ~97% of a round's wall clock.
    #
    # n_jobs=1 (the default) takes the historic INLINE path, byte for byte ---
    # not a one-worker pool --- so an existing run/trace is completely unaffected.
    # Above 1 the parent still runs admission and every append in candidate
    # order; only the pure backtest leaves the process, and the results are folded
    # back in ascending candidate order, so the prompts, the ledger, the trace and
    # the winner are identical to the serial run (see harness/parallel.py). Being
    # pure wall-clock, it is exempt from the resume guard and from a run's journal
    # identity (runner._RESUME_IGNORE), like the backoff knobs.
    n_jobs: int = 1

    # --- transient-failure backoff -----------------------------------------
    # A rate limit (429), a network blip or a CLI crash on one round is transient:
    # that round is skipped and the NEXT round re-asks the same question (see
    # runner.optimize --- the retry IS the next round, which is why the recorded
    # trace still holds exactly one completion per round). Without a pause those
    # remaining rounds fire back-to-back and a rate-limited search burns its whole
    # budget in about a second, so the wait grows exponentially with the number of
    # CONSECUTIVE failures: retry_backoff * 2**(k-1), capped at retry_max_delay.
    # Set either retry_backoff = 0 or retry_max_delay = 0 to disable the pause
    # entirely (what the test suite does). No jitter: this is a single-process
    # researcher's tool, not a fleet
    # that could stampede a shared endpoint, and a deterministic schedule is one
    # less thing to explain in a run log. Neither knob changes anything that is
    # COMPUTED --- only the pacing --- so both are exempt from the resume guard and
    # from a run's journal identity (see runner._RESUME_IGNORE).
    retry_backoff: float = 1.0                # seconds before the 1st retried round
    retry_max_delay: float = 30.0             # ceiling on the exponential growth

    # --- cross-run research memory -----------------------------------------
    # Read the persistent research journal (harness/memory/) into the proposer
    # prompt, and archive this run back into it when it finishes. This is the
    # researcher's accumulated CRAFT --- which mechanisms keep dying, which keep
    # earning their keep --- carried across runs the way a human's experience is.
    # It IS point-in-time gated, on exactly the same rule as an earlier date's OOS
    # score inside one walk (runner._prior_oos_closed): a journal lesson carries an
    # OOS grade, and a grade computable only from bars after T_n is look-ahead no
    # matter which run produced it, so a lesson is withheld until its own OOS
    # window has closed by this date's T_n (see proposer.journal_as_of).
    # What the gate cannot remove is the cost in independence: consulting the same
    # OOS window across many runs makes the reported Pass rate optimistic
    # (selection bias), so treat a Pass Line cleared after twenty journal-informed
    # runs as weaker evidence than a cold one.
    # The journal snapshot a run was given is recorded in its manifest, so replay
    # regenerates byte-identical prompts.
    memory: bool = True

    # --- misc --------------------------------------------------------------
    seed: int = 17
    max_tokens: int = 10000
    # Sampling control for providers that expose it. OpenCode 1.18.x accepts
    # this through the selected agent; providers without temperature support
    # may ignore it.
    temperature: float = 0.0
    # Codex CLI model-specific effort; None preserves the client's default.
    reasoning_effort: str | None = None
    logging: bool = True
    run_name: str = "demo"

    def __post_init__(self):
        require_contract(self.math_contract)
        from .providers.codex_controls import model_defaults, validate_config
        default_model = model_defaults(self.provider)
        if not self.model:
            self.model = default_model
        if self.reasoning_effort is not None:
            if self.reasoning_effort not in (
                    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
                raise ValueError(f"invalid reasoning_effort: {self.reasoning_effort!r}")
            if self.provider != "codex":
                raise ValueError("reasoning_effort currently requires provider='codex'")
        validate_config(self)
        if not _obj.is_valid(self.objective):
            raise ValueError(
                f"unknown objective '{self.objective}'; choose from {_obj.names()}"
            )
        if self.asset_class not in ASSET_PRESETS:
            raise ValueError(
                f"unknown asset_class '{self.asset_class}'; "
                f"choose from {list(ASSET_PRESETS)}"
            )
        # "all" means "every symbol this source can supply": for synthetic /
        # yfinance that is the asset class's basket (expanded here so the rest
        # of the code always sees a concrete list), but for ctx it is EVERY
        # underlying in the parquet, which only the data layer knows --- so the
        # sentinel is kept as ["all"] and expanded by data.load_ctx. Runs after
        # asset_class is validated, before the empty-universe check below. An
        # explicit symbol list is left untouched.
        if _is_all_sentinel(self.symbols):
            if self.data_source == "ctx":
                self.symbols = ["all"]
            else:
                self.symbols = list(ASSET_PRESETS[self.asset_class])
        elif isinstance(self.symbols, str):
            # A bare/comma string ("AAPL", "AAPL,MSFT", or CLI --set symbols=AAPL)
            # is normalised to a list so it can't slip past the empty-universe
            # guard below as a truthy str and crash deep in the data layer (pandas
            # would treat the str as a per-character column collection).
            self.symbols = [s.strip() for s in self.symbols.split(",") if s.strip()]
        if self.data_source not in ("synthetic", "yfinance", "ctx"):
            raise ValueError(
                f"unknown data_source '{self.data_source}'; "
                "choose from ['synthetic', 'yfinance', 'ctx']"
            )
        # An empty universe yields a zero-column panel: no cross-section to rank,
        # nothing to backtest. Catch it here rather than deep in the data layer.
        if not self.symbols:
            raise ValueError("symbols must be a non-empty universe (>= 1 symbol)")

        # A window override is only honoured when BOTH of its bounds are given
        # (see is_window/oos_window). Half a window silently falls back to the
        # clock, which almost always hides a config mistake -> reject it.
        if bool(self.is_start) != bool(self.is_end):
            raise ValueError(
                "is_start and is_end must be set together (or both left empty to "
                "use the research clock)"
            )
        if bool(self.oos_start) != bool(self.oos_end):
            raise ValueError(
                "oos_start and oos_end must be set together (or both left empty to "
                "use the research clock)"
            )

        # Numeric sanity: catch nonsensical knobs up front rather than producing
        # empty windows / silent no-ops deep in the backtest.
        def _require(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        _require(self.is_years > 0, f"is_years must be > 0, got {self.is_years}")
        _require(self.oos_days > 0, f"oos_days must be > 0, got {self.oos_days}")
        self.oos_mode = str(self.oos_mode).strip().lower()
        _require(self.oos_mode in {"fixed_days", "per_step"},
                 "oos_mode must be 'fixed_days' or 'per_step', "
                 f"got {self.oos_mode!r}")
        _require(self.max_iters >= 1, f"max_iters must be >= 1, got {self.max_iters}")
        _require(self.candidates_per_round >= 1,
                 f"candidates_per_round must be >= 1, got {self.candidates_per_round}")
        _require(self.n_jobs >= 1, f"n_jobs must be >= 1 (1 = no worker "
                                   f"processes), got {self.n_jobs}")
        # The one field that had no validation, and the most expensive to get
        # wrong --- a typo used to delete a finished run's ledger AND its LLM
        # trace before failing (see _check_frequency).
        _check_frequency(self.frequency)
        _require(self.min_is_days >= 1,
                 f"min_is_days must be >= 1, got {self.min_is_days}")
        _require(self.min_oos_days >= 1,
                 f"min_oos_days must be >= 1, got {self.min_oos_days}")
        # 0 is legal for both backoff knobs (either one disables the pause); negative is
        # not --- time.sleep would raise deep inside a failing search.
        _require(self.retry_backoff >= 0,
                 f"retry_backoff must be >= 0, got {self.retry_backoff}")
        _require(self.retry_max_delay >= 0,
                 f"retry_max_delay must be >= 0, got {self.retry_max_delay}")
        _require(self.cost_bps >= 0, f"cost_bps must be >= 0, got {self.cost_bps}")
        _require(self.gross > 0, f"gross must be > 0, got {self.gross}")
        _require(self.initial_cash > 0,
                 f"initial_cash must be > 0, got {self.initial_cash}")
        _require(self.annualization > 0,
                 f"annualization must be > 0, got {self.annualization}")
        self.selection_mode = str(self.selection_mode).strip().lower()
        _require(self.selection_mode in {"none", "top_q", "top_k"},
                 "selection_mode must be 'none', 'top_q' or 'top_k', "
                 f"got {self.selection_mode!r}")
        _require(0.0 < self.top_q <= 1.0,
                 f"top_q must be > 0 and <= 1, got {self.top_q}")
        _require(self.top_k >= 1,
                 f"top_k must be >= 1, got {self.top_k}")
        _require(self.hold_every >= 1,
                 f"hold_every must be >= 1, got {self.hold_every}")
        _require(self.rebalance_every >= 1,
                 f"rebalance_every must be >= 1, got {self.rebalance_every}")
        _require(self.warmup >= 0,
                 f"warmup must be >= 0, got {self.warmup}")
        _require(self.temperature == self.temperature
                 and abs(self.temperature) != float("inf")
                 and self.temperature >= 0,
                 f"temperature must be finite and >= 0, got {self.temperature}")
        _require(self.max_tokens >= 1,
                 f"max_tokens must be >= 1, got {self.max_tokens}")
        # pass_line may legitimately be negative (e.g. maxdd Pass Line -0.2), so
        # only rule out non-finite values.
        _require(self.pass_line == self.pass_line and abs(self.pass_line) != float("inf"),
                 f"pass_line must be finite, got {self.pass_line}")

    # --- derived date windows ---------------------------------------------
    def is_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """[start, end] in-sample optimisation window."""
        if self.is_start and self.is_end:
            return pd.Timestamp(self.is_start), pd.Timestamp(self.is_end)
        tn = pd.Timestamp(self.research_date)
        return tn - pd.DateOffset(years=self.is_years), tn

    def oos_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """[start, end] held-out window (validate.py guarantees start > IS end).

        ``oos_days`` counts TRADING days (business days), matching the daily-bar
        clock the backtest runs on, so the horizon isn't shortened by the ~2/7 of
        calendar days that are weekends (a plain ``Timedelta(days=oos_days)``
        would land ~30% short in bar count).
        """
        if self.oos_start and self.oos_end:
            return pd.Timestamp(self.oos_start), pd.Timestamp(self.oos_end)
        tn = pd.Timestamp(self.research_date)
        return tn, tn + pd.tseries.offsets.BDay(self.oos_days)

    def backtest_weight_config(self) -> dict:
        """Return portfolio-construction controls for BacktestEngine.Position."""
        return {
            "selection_mode": self.selection_mode,
            "top_q": float(self.top_q),
            "top_k": int(self.top_k),
            "hold_every": int(self.hold_every),
            "rebalance_every": int(self.rebalance_every),
        }

    # --- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        if not isinstance(d, dict):
            raise TypeError(
                f"configuration must be a JSON object, got {type(d).__name__}"
            )
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_recorded_dict(cls, data: dict) -> "RunConfig":
        """Recorded runs must contain the complete current schema, without defaults."""
        if not isinstance(data, dict):
            raise ValueError("CONFIG_SCHEMA_MISMATCH: recorded configuration must be an object")
        require_contract(data.get("math_contract"))
        missing = {f.name for f in fields(cls)} - set(data)
        if missing:
            raise ValueError(f"CONFIG_SCHEMA_MISMATCH: incomplete recorded configuration: {sorted(missing)}")
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
