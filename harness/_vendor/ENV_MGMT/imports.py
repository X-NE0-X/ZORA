"""
**`from ENV_MGMT.imports import *`**

---

- import `csv`
- import `copy`
- import `time`
- import `json`
- import `talib`
- import `duckdb`
- import `hashlib`
- import `fnmatch`
- import `datetime`
- import `functools`
- import `threading`
- import `itertools`
- import `subprocess`
- import `matplotlib`
- import `cloudpickle`
- import `numpy` as `np`
- import `pandas` as `pd`
- import `matplotlib.pyplot` as `plt`
- import `warnings`; warnings.filterwarnings("ignore")
- import `inspect`, `logging`, `tempfile`, `traceback`, `pickle`, `cloudpickle`, `multiprocessing`
- import `os`, `io`, `re`, `ast`, `sys`, `math`, `code`, `queue`, `random`, `builtins`, `argparse`, `textwrap`, `pathlib`
- import `importlib`, `importlib.util`

- from `numba` import `njit`
- from `numpy.typing` import `NDArray`
- from `numpy` import `float64`, `float32`
- from `collections` import `OrderedDict`
- from `collections.abc` import `Iterable`
- from `dataclasses` import `asdict`, `dataclass`, `replace`, `field`
- from `multiprocessing` import `Manager`, `freeze_support`
- from `logging.handlers` import `QueueHandler`, `QueueListener`
- from `concurrent.futures` import `ProcessPoolExecutor`, `ThreadPoolExecutor`, `as_completed`, `BrokenExecutor`
- from `typing` import `Dict`, `List`, `Tuple`, `Optional`, `Set`, `Union`, `Any`, `Callable`, `Type`, `Literal`, `Hashable`, `Mapping`, `Sequence`, `cast`, `Iterator`, `TYPE_CHECKING`
- from `rich.progress` import (
    `Progress`,
    `TextColumn`,
    `BarColumn`,
    `MofNCompleteColumn`,
    `TimeElapsedColumn`,
    `TimeRemainingColumn`,
)


Will Try:
- import `vectorbt` as `vbt`
- from `vectorbt.utils.array_api` import `to_1d`   (local numpy fallback if unavailable)
"""

import csv
import copy
import time
import json
import talib
import duckdb
import hashlib
import fnmatch
import datetime
import functools
import threading
import itertools
import subprocess
import matplotlib
import cloudpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import inspect, logging, tempfile, traceback, pickle, cloudpickle, multiprocessing
import os, io, re, ast, sys, math, code, queue, random, builtins, argparse, textwrap, pathlib
import importlib, importlib.util

from numba import njit
from numpy.typing import NDArray
from numpy import float64, float32
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace, field
from multiprocessing import Manager, freeze_support
from logging.handlers import QueueHandler, QueueListener
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, BrokenExecutor
from typing import Dict, List, Tuple, Optional, Set, Union, Any, Callable, Type, Literal, Hashable, Mapping, Sequence, cast, Iterator, TYPE_CHECKING
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


try:
    import vectorbt as vbt

except Exception:
    vbt = None
    warnings.warn(
        "[WARNING] vectorbt import failed; vectorbt features are disabled.",
        RuntimeWarning,
    )


try:
    from vectorbt.utils.array_api import to_1d

except Exception:

    def to_1d(arr):
        """Local fallback for vectorbt's `to_1d`: flatten any array-like to a 1-D numpy vector."""

        return np.asarray(arr).reshape(-1)
