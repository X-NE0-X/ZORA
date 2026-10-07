from ENV_MGMT.imports import *



def _strategy_pack_for_transport(backtest_strategy: Any) -> Any:

    if isinstance(backtest_strategy, memoryview):

        return bytes(backtest_strategy)

    if isinstance(backtest_strategy, bytearray):

        return bytes(backtest_strategy)

    if isinstance(backtest_strategy, bytes):

        return backtest_strategy

    if callable(backtest_strategy):

        try:

            return cloudpickle.dumps(backtest_strategy)

        except Exception:

            return backtest_strategy


    return backtest_strategy


def _strategy_resolve_callable(backtest_strategy: Any) -> Callable[..., Any]:

    strategy_obj = backtest_strategy

    if isinstance(strategy_obj, memoryview):
        strategy_obj = bytes(strategy_obj)

    if isinstance(strategy_obj, bytearray):
        strategy_obj = bytes(strategy_obj)

    if isinstance(strategy_obj, bytes):
        strategy_obj = cloudpickle.loads(strategy_obj)

    if not callable(strategy_obj):

        raise TypeError(f"[WARNING] backtest_strategy is not callable: {type(strategy_obj)!r}")


    return strategy_obj


def _is_signature_mismatch_type_error(exc: BaseException) -> bool:

    if not isinstance(exc, TypeError):

        return False

    text = str(exc).lower()
    hints = (
        "signature mismatch",
        "unexpected keyword argument",
        "required positional argument",
        "missing 1 required positional argument",
        "takes",
        "positional argument",
        "got multiple values for argument",
    )

    return any(h in text for h in hints)




# Settings
#----------------------------------------------------------------------------------------
def plot_display_set() -> None:
    """
    ### What It Does
    Configures the notebook display defaults used by plotting helpers.

    #### Responsibility
    Sets pandas, numpy, and matplotlib display behavior for research inspection.

    #### How To Use
    Call it once near the start of an interactive research session.

    #### Usage Example
    `result = plot_display_set(...)`
    """

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["figure.dpi"] = 250

    plt.rcParams['figure.facecolor'] = '#131722'
    plt.rcParams['figure.edgecolor'] = '#131722'
    plt.rcParams['axes.facecolor'] = '#131722'
    plt.rcParams['axes.labelcolor'] = 'white'

    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['xtick.labelcolor'] = 'white'
    plt.rcParams['ytick.labelcolor'] = 'white'

    plt.rcParams['grid.color'] = 'grey'
    plt.rcParams['grid.linestyle'] = '--'

    plt.rcParams['lines.color'] = 'white'
    plt.rcParams['patch.edgecolor'] = '#EFD084'
    plt.rcParams['patch.facecolor'] = '#EFD084'

    print('Matplotlib......Configured')


def output_display_set() -> None:
    """
    ### What It Does
    Configures console and DataFrame output visibility for backtest diagnostics.

    #### Responsibility
    Keeps display width and truncation settings predictable during reporting.

    #### How To Use
    Call it before running workflows that print large tables or diagnostics.

    #### Usage Example
    `result = output_display_set(...)`
    """

    pd.set_option('display.max_rows', None)

    try:
        stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)

        if callable(stdout_reconfigure):
            stdout_reconfigure(encoding = "utf-8", errors = "replace")

        stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)

        if callable(stderr_reconfigure):
            stderr_reconfigure(encoding = "utf-8", errors = "replace")

    except Exception:
        pass

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    np.set_printoptions(suppress = True, precision = 6, floatmode = "fixed")

    print('Output Display Options......Set')


def print_override_set() -> None:
    """
    ### What It Does
    Installs the module's print override with optional log routing.

    #### Responsibility
    Routes printed output through the engine's reporting hooks while preserving normal console behavior.

    #### How To Use
    Call it before long-running protocols that need synchronized console and log output.

    #### Usage Example
    `result = print_override_set(...)`
    """

    _real_print = builtins.print

    def print_override(*args: Any, **kwargs: Any) -> None:
        """
        ### What It Does
        Prints through the active override path.

        #### Responsibility
        Mirrors messages to configured output sinks without making callers manage handlers.

        #### How To Use
        Use it like `print(...)` inside this module.

        #### Key Parameters In Practice
        - `*args`
          - Additional positional inputs forwarded to the underlying callable/dispatcher. Use only when the wrapped API explicitly expects positional values.
          - Expected shape/type: `Any`.
        - `**kwargs`
          - Additional keyword arguments forwarded to the lower-level dispatcher. Prefer explicit named parameters in notebooks when the run must be reproducible.
          - Expected shape/type: `Any`.

        #### Usage Example
        `result = print_override(...)`

        ---

        ### Parameters
        - `*args`: **Any**.
        - `**kwargs`: **Any**.
        """

        if multiprocessing.current_process().name == "MainProcess":
            _real_print(*args, **kwargs)


    builtins.print = print_override

    print('Print Override......Set')


def multi_log_set_top() -> Tuple[Any, list]:
    """
    ### What It Does
    Configures top-level multi-process logging state.

    #### Responsibility
    Initializes shared queue and path information used by worker logging.

    #### How To Use
    Call it before dispatching multiprocessing backtest or tuning workers.

    #### Usage Example
    `result = multi_log_set_top(...)`

    ---

    ### Returns
    - `result`: **Tuple[Any, list]**.
    """

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    print('Top Multiprocessing Logging......Set')


    return root, root.handlers


def multi_log_set_main(log_dir: str, log_filename: str) -> Tuple[QueueListener, Any, Any, str]:
    """
    ### What It Does
    Configures main-process logging for multi-worker protocols.

    #### Responsibility
    Keeps main and worker log messages aligned under the same output contract.

    #### How To Use
    Call it from the parent process before worker execution begins.

    #### Key Parameters In Practice
    - `log_dir`
      - Directory for run logs and runner outputs. Point it at `ArchiveDeck/ComputerLog` or a run-specific child folder so outputs do not scatter across notebooks.
      - Expected shape/type: `str`.
    - `log_filename`
      - Log file name under `log_dir`. Use a stable run id or protocol id so multi-worker logs can be traced to the exact optimizer run.
      - Expected shape/type: `str`.


    #### Usage Example
    `result = multi_log_set_main(...)`

    ---

    ### Parameters
    - `log_dir`: **str**.
    - `log_filename`: **str**.

    ---

    ### Returns
    - `result`: **Tuple[QueueListener, Any, Any, str]**.
    """

    multiprocessing.freeze_support()

    log_mgr = Manager()
    log_queue = log_mgr.Queue()

    LOGFMT = "%(asctime)s %(levelname)s: %(message)s"
    handler = QueueHandler(log_queue)
    root = logging.getLogger()
    root.addHandler(handler)

    os.makedirs(log_dir, exist_ok = True)
    log_path = os.path.join(log_dir, log_filename)

    file_hdl = logging.FileHandler(log_path, mode = "w", encoding = "utf-8")
    file_hdl.setFormatter(logging.Formatter(LOGFMT))
    file_hdl.setLevel(logging.INFO)

    listener = QueueListener(log_queue, file_hdl)
    listener.start()

    console_hdl = logging.StreamHandler(sys.stdout)
    console_hdl.setFormatter(logging.Formatter(LOGFMT))
    console_hdl.setLevel(logging.WARNING)
    logging.getLogger().addHandler(console_hdl)

    print('Main Multiprocessing Logging......Set')


    return listener, log_queue, log_mgr, log_path
