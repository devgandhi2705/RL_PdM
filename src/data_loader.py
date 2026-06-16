"""
data_loader.py
==============
Load and label the PRONOSTIA (FEMTO-ST) bearing dataset.

Dataset layout (raw_dir)
------------------------
raw_dir/
  Bearing1_1/
    acc_00001.csv
    acc_00002.csv
    ...
  Bearing1_2/
    ...
  Bearing2_1/
    ...

Each CSV has 8 columns (no header):
  col 0 – hour
  col 1 – min
  col 2 – sec
  col 3 – microsec
  col 4 – horiz_acc   ← used
  col 5 – vert_acc    ← used
  col 6 – (unused)
  col 7 – (unused)

Every file contains exactly SAMPLES_PER_FILE rows (2560 by default).

References
----------
Nectoux et al., "PRONOSTIA: An Experimental Platform for Bearings Accelerated
Life Test", IEEE ICKDDM, 2012.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (overridable via config)
# ---------------------------------------------------------------------------
SAMPLES_PER_FILE: int = 2560
N_CHANNELS: int = 2          # horiz_acc, vert_acc
MAX_RUL: int = 125           # piece-wise linear health index cap
HORIZ_COL: int = 4
VERT_COL: int = 5

# Bearings available on disk (6 total in this dataset).
# TRAIN_BEARINGS are used to train the RUL predictor (supervised).
# TEST_BEARINGS are held out from RUL predictor training for unbiased eval.
# The RL agent uses ALL_BEARINGS for maximum episode diversity —
# it learns a maintenance policy, not an RUL predictor.
TRAIN_BEARINGS: List[str] = ["1_1", "1_2", "2_1", "2_2", "3_1"]
TEST_BEARINGS:  List[str] = ["3_2"]
ALL_BEARINGS:   List[str] = TRAIN_BEARINGS + TEST_BEARINGS

# Bearings grouped by operating condition (only bearings present on disk)
CONDITION_MAP: Dict[int, List[str]] = {
    1: ["1_1", "1_2"],
    2: ["2_1", "2_2"],
    3: ["3_1", "3_2"],
}


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Return parsed YAML config.  Falls back to built-in defaults if the
    file is not found so the module stays usable in isolation."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("config.yaml not found; using built-in defaults.")
        return {}
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def _sorted_acc_files(bearing_dir: Path) -> List[Path]:
    """Return acc_NNNNN.csv files in the bearing directory, sorted numerically."""
    files = sorted(
        bearing_dir.glob("acc_*.csv"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    if len(files) > 1:
        indices = [int(re.search(r"(\d+)", p.stem).group(1)) for p in files]
        assert indices == sorted(indices) and len(set(indices)) == len(indices), (
            f"acc_*.csv files in {bearing_dir} are not in strict ascending order: "
            f"{[p.name for p in files[:5]]} …"
        )
    return files


def load_bearing(
    bearing_dir: Path,
    samples_per_file: int = SAMPLES_PER_FILE,
) -> np.ndarray:
    """Load all accelerometer CSV files for a single bearing.

    Parameters
    ----------
    bearing_dir:
        Path to a folder such as ``raw/Bearing1_1``.
    samples_per_file:
        Expected number of rows per CSV.  If a file has fewer rows it is
        zero-padded; extra rows are truncated.

    Returns
    -------
    np.ndarray
        Shape ``(N_files, samples_per_file, 2)`` — axis-2 is
        ``[horiz_acc, vert_acc]``.  dtype = float32.
    """
    acc_files = _sorted_acc_files(bearing_dir)
    if not acc_files:
        raise FileNotFoundError(
            f"No acc_*.csv files found in {bearing_dir}"
        )

    n_files = len(acc_files)
    data = np.zeros((n_files, samples_per_file, N_CHANNELS), dtype=np.float32)

    for i, fp in enumerate(acc_files):
        try:
            df = pd.read_csv(fp, header=None, usecols=[HORIZ_COL, VERT_COL])
        except Exception as exc:
            raise RuntimeError(f"Failed to read {fp}: {exc}") from exc

        arr = df.values.astype(np.float32)  # shape (rows, 2)
        rows = min(arr.shape[0], samples_per_file)
        data[i, :rows, :] = arr[:rows, :]
        if rows < samples_per_file:
            logger.debug("File %s has %d rows (expected %d); zero-padded.",
                         fp.name, rows, samples_per_file)

    logger.debug("Loaded bearing %s — %d files.", bearing_dir.name, n_files)
    return data


def compute_rul_labels(n_files: int, max_rul: int = MAX_RUL) -> np.ndarray:
    """Compute piece-wise linear RUL labels for a single bearing run.

    RUL at file index *i* equals ``(n_files - i - 1)`` clipped to
    ``max_rul``.  This is the standard PRONOSTIA health index definition:
    full health (value = max_rul) during most of the run, decreasing
    linearly only in the final ``max_rul`` measurement windows.

    Parameters
    ----------
    n_files:
        Total number of measurement windows (CSV files) for this bearing.
    max_rul:
        Maximum RUL cap.  Defaults to 125 (PRONOSTIA standard).

    Returns
    -------
    np.ndarray
        Shape ``(n_files,)`` of float32 RUL values in ``[0, max_rul]``.
    """
    indices = np.arange(n_files, dtype=np.float32)
    rul = (n_files - 1 - indices).clip(0, max_rul)
    return rul


def load_pronostia(
    data_dir: str | Path,
    samples_per_file: int = SAMPLES_PER_FILE,
    max_rul: int = MAX_RUL,
    bearing_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the full PRONOSTIA dataset from *data_dir*.

    Scans *data_dir* for sub-folders matching ``Bearing{cond}_{idx}`` and
    loads every acc_*.csv file found inside each one.

    Parameters
    ----------
    data_dir:
        Root directory that contains ``Bearing1_1/``, ``Bearing1_2/``, etc.
    samples_per_file:
        Rows expected per CSV file (default 2560).
    max_rul:
        RUL cap for the piece-wise linear health index (default 125).
    bearing_ids:
        Optional list of bearing IDs to load (e.g. ``["1_1", "3_2"]``).
        When *None* all bearings found on disk are loaded.

    Returns
    -------
    dict
        Nested dict with structure::

            {
              "1_1": {
                  "data":  np.ndarray,  # (N_files, 2560, 2), float32
                  "rul":   np.ndarray,  # (N_files,),         float32
              },
              "1_2": { ... },
              ...
            }
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir does not exist: {data_dir}")

    # Discover bearing folders
    pattern = re.compile(r"^Bearing(\d+)_(\d+)$", re.IGNORECASE)
    found: Dict[str, Path] = {}
    for entry in sorted(data_dir.iterdir()):
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                bid = f"{m.group(1)}_{m.group(2)}"
                found[bid] = entry

    if not found:
        raise FileNotFoundError(
            f"No Bearing*_* directories found in {data_dir}"
        )

    if bearing_ids is not None:
        missing = set(bearing_ids) - set(found)
        if missing:
            logger.warning("Requested bearings not found on disk: %s", missing)
        found = {k: v for k, v in found.items() if k in bearing_ids}

    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    for bid, bdir in found.items():
        logger.info("Loading bearing %s from %s …", bid, bdir)
        data = load_bearing(bdir, samples_per_file=samples_per_file)
        rul = compute_rul_labels(data.shape[0], max_rul=max_rul)
        dataset[bid] = {"data": data, "rul": rul}

    logger.info("Loaded %d bearing(s) from %s.", len(dataset), data_dir)
    return dataset


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def get_condition(bearing_id: str) -> int:
    """Return the operating condition (1, 2, or 3) for a bearing ID."""
    cond_digit = int(bearing_id.split("_")[0])
    return cond_digit


def split_by_condition(
    dataset: Dict[str, Dict[str, np.ndarray]],
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """Group bearings in *dataset* by operating condition.

    Returns
    -------
    dict
        ``{1: {"1_1": {...}, "1_2": {...}}, 2: {...}, 3: {...}}``
    """
    by_cond: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {1: {}, 2: {}, 3: {}}
    for bid, arrays in dataset.items():
        cond = get_condition(bid)
        by_cond.setdefault(cond, {})[bid] = arrays
    return by_cond


def train_test_split_pronostia(
    dataset: Dict[str, Dict[str, np.ndarray]],
    train_ids: Optional[List[str]] = None,
    test_ids: Optional[List[str]] = None,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
]:
    """Split *dataset* into train and test subsets.

    Uses the standard PRONOSTIA split when *train_ids* / *test_ids* are
    *None*:
    - Train: 1_1, 1_2, 2_1, 2_2, 3_1
    - Test:  everything else

    Parameters
    ----------
    dataset:
        Output of :func:`load_pronostia`.
    train_ids:
        Override the default training bearing IDs.
    test_ids:
        Override the default test bearing IDs.  When *None* all bearings
        not in *train_ids* are treated as test.

    Returns
    -------
    (train_dataset, test_dataset)
        Both have the same nested-dict structure as the input.
    """
    if train_ids is None:
        train_ids = TRAIN_BEARINGS
    if test_ids is None:
        test_ids = [bid for bid in dataset if bid not in train_ids]

    train = {bid: dataset[bid] for bid in train_ids if bid in dataset}
    test = {bid: dataset[bid] for bid in test_ids if bid in dataset}

    logger.info(
        "Split — train: %s | test: %s",
        sorted(train.keys()),
        sorted(test.keys()),
    )
    return train, test


# ---------------------------------------------------------------------------
# Quick-access convenience wrapper
# ---------------------------------------------------------------------------

def build_dataset(
    config_path: str | Path = "config.yaml",
    bearing_ids: Optional[List[str]] = None,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
]:
    """One-call helper: load config → load data → split.

    Parameters
    ----------
    config_path:
        Path to ``config.yaml`` (default searches current directory).
    bearing_ids:
        Restrict which bearings are loaded.  *None* loads all.

    Returns
    -------
    (train_dataset, test_dataset)
        Dictionaries as returned by :func:`train_test_split_pronostia`.
    """
    cfg = load_config(config_path)
    ds_cfg = cfg.get("dataset", {})

    raw_dir = Path(ds_cfg.get("raw_dir", "data/raw"))
    spf = int(ds_cfg.get("samples_per_file", SAMPLES_PER_FILE))
    max_rul = int(ds_cfg.get("max_rul", MAX_RUL))
    train_ids = ds_cfg.get("train_bearings", None)
    test_ids = ds_cfg.get("test_bearings", None)

    dataset = load_pronostia(
        raw_dir,
        samples_per_file=spf,
        max_rul=max_rul,
        bearing_ids=bearing_ids,
    )
    return train_test_split_pronostia(dataset, train_ids=train_ids, test_ids=test_ids)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Make the project root importable regardless of how this script is invoked
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.device import get_device

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    get_device()  # print device banner once at startup
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    raw_path = Path(data_dir)
    if not raw_path.is_dir():
        print(f"[INFO] Directory '{data_dir}' does not exist yet.")
        print("       Place the PRONOSTIA bearing folders (Bearing1_1/, Bearing1_2/, …)")
        print(f"       inside '{data_dir}' and re-run this script.")
        sys.exit(0)

    bearing_dirs = [
        e for e in raw_path.iterdir()
        if e.is_dir() and re.match(r"^Bearing\d+_\d+$", e.name, re.IGNORECASE)
    ]
    if not bearing_dirs:
        print(f"[INFO] No Bearing*_* sub-folders found in '{data_dir}'.")
        print("       Expected layout:  data/raw/Bearing1_1/acc_00001.csv  …")
        print(f"       Folders present:  {[e.name for e in raw_path.iterdir() if e.is_dir()] or '(none)'}")
        sys.exit(0)

    ds = load_pronostia(data_dir)
    for bid, arrays in sorted(ds.items()):
        d, r = arrays["data"], arrays["rul"]
        print(f"Bearing {bid}: data={d.shape}  rul={r.shape}  "
              f"rul range=[{r.min():.0f}, {r.max():.0f}]")
