from ENV_MGMT.imports import *


# Project root, resolved locally so this module works standalone or via the package facade
#----------------------------------------------------------------------------------------
# Calendars.py lives at <project>/harness/_vendor/BacktestEngine/, so parents[3] is the
# harness project root. Upstream searched the parents for a monorepo root instead; that
# ancestor does not exist here, so the search always fell through to this same path.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]





# Back-test Data
#----------------------------------------------------------------------------------------
def default_holidays() -> List[datetime.date]:
    """
    ### What It Does
    Returns the default holiday calendar used by the backtests.

    #### Responsibility
    Provides a shared market-closure list when callers do not supply one.

    #### How To Use
    Call it while building signal or backtest configs without custom holidays.

    #### Usage Example
    `result = default_holidays(...)`

    ---

    ### Returns
    - `result`: **List[datetime.date]**.
    """

    csv_path = _PROJECT_ROOT / "data" / "holidays_nyse.csv"

    holidays: List[datetime.date] = []

    if csv_path.exists():

        with open(csv_path, newline = "", encoding = "utf-8") as fp:
            reader = csv.DictReader(fp)

            for row in reader:

                try:
                    raw_date = (
                        row.get("date")
                        or row.get("Datetime")
                        or row.get("datetime")
                        or row.get("Date")
                    )

                    if raw_date is None:

                        continue

                    date_token = str(raw_date).strip().split()[0].replace("/", "-")
                    parts = date_token.split("-")
                    holidays.append(datetime.date(int(parts[0]), int(parts[1]), int(parts[2])))

                except Exception as exc:

                    logging.warning("[WARNING] skipped unparseable NYSE holiday row %r: %s", raw_date, exc)

                    continue

        if not holidays:
            logging.warning(
                            "[WARNING] NYSE holidays CSV at %s parsed to ZERO usable rows "
                            "(date-format mismatch? every row was skipped); calendar is empty.",
                            csv_path,
                            )

    else:
        logging.warning(
                        "[WARNING] NYSE holidays CSV not found at %s; falling back to a built-in static list "
                        "covering 2024-2026 (observed NYSE full-day closures). Supply an up-to-date "
                        "holidays_nyse.csv for calendars beyond 2026.",
                        csv_path,
                        )
        # Observed NYSE full-day closures (weekend-shifted). NOT a generic/CN calendar:
        # New Year, MLK, Washington's Birthday, Good Friday, Memorial, Juneteenth,
        # Independence, Labor, Thanksgiving, Christmas.
        holidays = [
            # 2024
            datetime.date(2024, 1, 1),  datetime.date(2024, 1, 15), datetime.date(2024, 2, 19),
            datetime.date(2024, 3, 29), datetime.date(2024, 5, 27), datetime.date(2024, 6, 19),
            datetime.date(2024, 7, 4),  datetime.date(2024, 9, 2),  datetime.date(2024, 11, 28),
            datetime.date(2024, 12, 25),
            # 2025
            datetime.date(2025, 1, 1),  datetime.date(2025, 1, 20), datetime.date(2025, 2, 17),
            datetime.date(2025, 4, 18), datetime.date(2025, 5, 26), datetime.date(2025, 6, 19),
            datetime.date(2025, 7, 4),  datetime.date(2025, 9, 1),  datetime.date(2025, 11, 27),
            datetime.date(2025, 12, 25),
            # 2026
            datetime.date(2026, 1, 1),  datetime.date(2026, 1, 19), datetime.date(2026, 2, 16),
            datetime.date(2026, 4, 3),  datetime.date(2026, 5, 25), datetime.date(2026, 6, 19),
            datetime.date(2026, 7, 3),  datetime.date(2026, 9, 7),  datetime.date(2026, 11, 26),
            datetime.date(2026, 12, 25),
        ]

    if holidays and max(holidays) < datetime.date.today():

        logging.warning(
                        "[WARNING] holiday calendar looks stale: latest entry %s is before today %s.",
                        max(holidays),
                        datetime.date.today(),
                        )


    return holidays


def default_macro_release_dates() -> List[datetime.date]:
    """
    ### What It Does
    Returns default macro release dates used by signal-blocking logic.

    #### Responsibility
    Provides shared event dates for preventing exposure around macro releases.

    #### How To Use
    Call it when weight construction needs a macro calendar but config omits one.

    #### Usage Example
    `result = default_macro_release_dates(...)`

    ---

    ### Returns
    - `result`: **List[datetime.date]**.
    """

    macro_dates = [

        datetime.date(2024, 1, 5), datetime.date(2024, 1, 11), datetime.date(2024, 1, 25),
        datetime.date(2024, 2, 2), datetime.date(2024, 2, 13), datetime.date(2024, 2, 28),
        datetime.date(2024, 3, 8), datetime.date(2024, 3, 12), datetime.date(2024, 3, 28),
        datetime.date(2024, 4, 5), datetime.date(2024, 4, 10), datetime.date(2024, 4, 25),
        datetime.date(2024, 5, 3), datetime.date(2024, 5, 15), datetime.date(2024, 5, 30),
        datetime.date(2024, 6, 7), datetime.date(2024, 6, 12), datetime.date(2024, 6, 27),
        datetime.date(2024, 7, 5), datetime.date(2024, 7, 11), datetime.date(2024, 7, 25),
        datetime.date(2024, 8, 2), datetime.date(2024, 8, 14), datetime.date(2024, 8, 29),
        datetime.date(2024, 9, 6), datetime.date(2024, 9, 11), datetime.date(2024, 9, 26),
        datetime.date(2024, 10, 4), datetime.date(2024, 10, 10), datetime.date(2024, 10, 30),
        datetime.date(2024, 11, 1), datetime.date(2024, 11, 13), datetime.date(2024, 11, 27),
        datetime.date(2024, 12, 6), datetime.date(2024, 12, 11), datetime.date(2024, 12, 19),
        datetime.date(2025, 1, 10), datetime.date(2025, 1, 15), datetime.date(2025, 1, 30),
        datetime.date(2025, 2, 7), datetime.date(2025, 2, 12), datetime.date(2025, 2, 27),
        datetime.date(2025, 3, 7), datetime.date(2025, 3, 12), datetime.date(2025, 3, 27),
        datetime.date(2025, 4, 4), datetime.date(2025, 4, 10), datetime.date(2025, 4, 30),
        datetime.date(2025, 5, 2), datetime.date(2025, 5, 13), datetime.date(2025, 5, 29),
        datetime.date(2025, 6, 6), datetime.date(2025, 6, 11), datetime.date(2025, 6, 26),
        datetime.date(2025, 7, 3), datetime.date(2025, 7, 15), datetime.date(2025, 7, 30),
        datetime.date(2025, 8, 1), datetime.date(2025, 8, 12), datetime.date(2025, 8, 28),
        datetime.date(2025, 9, 5), datetime.date(2025, 9, 11), datetime.date(2025, 9, 25),
        datetime.date(2025, 10, 3), datetime.date(2025, 10, 15), datetime.date(2025, 10, 30),
        datetime.date(2025, 11, 7), datetime.date(2025, 11, 13), datetime.date(2025, 11, 26),
        datetime.date(2025, 12, 5), datetime.date(2025, 12, 10), datetime.date(2025, 12, 18),
    ]

    if macro_dates and max(macro_dates) < datetime.date.today():

        logging.warning(
                        "[WARNING] macro-release calendar looks stale: latest entry %s is before today %s.",
                        max(macro_dates),
                        datetime.date.today(),
                        )


    return macro_dates
