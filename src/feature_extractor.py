"""
feature_extractor.py
====================
Sliding-window feature extraction for PRONOSTIA raw vibration signals.

Input
-----
raw_array : np.ndarray, shape (N_files, 2560, 2)
    Axis-0: measurement files (timesteps).
    Axis-1: 2560 samples per file.
    Axis-2: channel 0 = horizontal acc, channel 1 = vertical acc.

Output
------
features : np.ndarray, shape (N_files, 32), float32

Feature layout (32 total = 16 per channel × 2 channels)
---------------------------------------------------------
Per channel — time-domain (indices 0-11 for H, 16-27 for V):
   0  RMS
   1  Peak (max |x|)
   2  Crest factor       = peak / RMS
   3  Kurtosis           (Fisher, bias=True)
   4  Skewness           (bias=True)
   5  Variance
   6  Mean absolute value
   7  Peak-to-peak       = max(x) - min(x)
   8  Shape factor       = RMS / mean|x|
   9  Impulse factor     = peak / mean|x|
  10  Clearance factor   = peak / (mean(sqrt|x|))^2
  11  Histogram entropy  = -sum(p * log(p)), 10-bin normalised histogram

Per channel — frequency-domain (indices 12-15 for H, 28-31 for V):
  12  Frequency centre   = sum(f * |X|) / sum(|X|)
  13  RMS frequency      = sqrt( sum(f^2 * |X|^2) / sum(|X|^2) )
  14  Frequency variance = sum( (f - fc)^2 * |X|^2 ) / sum(|X|^2)
  15  Spectral entropy   = -sum(p * log(p)), p = |X|^2 / sum(|X|^2)

Sampling frequency: 25 600 Hz (PRONOSTIA standard).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# PRONOSTIA accelerometer sampling frequency (Hz)
FS: float = 25_600.0

# Total number of features per timestep
N_FEATURES: int = 32
N_TIME_FEATURES: int = 12  # per channel
N_FREQ_FEATURES: int = 4   # per channel
N_CHANNELS: int = 2

# Feature names for reference / DataFrame columns
FEATURE_NAMES: List[str] = [
    f"{ch}_{name}"
    for ch in ("H", "V")
    for name in (
        "rms", "peak", "crest_factor", "kurtosis", "skewness", "variance",
        "mean_abs", "peak_to_peak", "shape_factor", "impulse_factor",
        "clearance_factor", "hist_entropy",
        "freq_centre", "rms_freq", "freq_variance", "spectral_entropy",
    )
]


# ---------------------------------------------------------------------------
# Time-domain features (12 per channel)
# ---------------------------------------------------------------------------

def _time_domain_features(x: np.ndarray) -> np.ndarray:
    """Compute 12 time-domain features for a 1-D signal.

    Parameters
    ----------
    x : np.ndarray, shape (N,)

    Returns
    -------
    np.ndarray, shape (12,), float32
    """
    eps = 1e-10

    rms         = np.sqrt(np.mean(x ** 2))
    peak        = np.max(np.abs(x))
    mean_abs    = np.mean(np.abs(x))
    sqrt_abs    = np.mean(np.sqrt(np.abs(x)))

    crest_factor    = peak / (rms + eps)
    shape_factor    = rms / (mean_abs + eps)
    impulse_factor  = peak / (mean_abs + eps)
    clearance_factor = peak / (sqrt_abs ** 2 + eps)

    kurt        = float(scipy_kurtosis(x, fisher=True, bias=True))
    skewness    = float(scipy_skew(x, bias=True))
    variance    = float(np.var(x, ddof=0))
    peak_to_peak = float(np.max(x) - np.min(x))

    # Histogram entropy over 10 bins
    counts, _ = np.histogram(x, bins=10)
    counts     = counts.astype(np.float64) + eps
    p          = counts / counts.sum()
    hist_entropy = float(-np.sum(p * np.log(p)))

    return np.array([
        rms, peak, crest_factor, kurt, skewness, variance,
        mean_abs, peak_to_peak, shape_factor, impulse_factor,
        clearance_factor, hist_entropy,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Frequency-domain features (4 per channel)
# ---------------------------------------------------------------------------

def _freq_domain_features(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Compute 4 frequency-domain features for a 1-D signal.

    Uses the one-sided FFT magnitude spectrum.

    Parameters
    ----------
    x  : np.ndarray, shape (N,)
    fs : float — sampling frequency in Hz

    Returns
    -------
    np.ndarray, shape (4,), float32
        [freq_centre, rms_freq, freq_variance, spectral_entropy]
    """
    n      = len(x)
    mag    = np.abs(np.fft.rfft(x))          # |X(f)|
    power  = mag ** 2                         # |X(f)|^2
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)  # frequency axis (Hz)

    eps       = 1e-30
    mag_sum   = mag.sum()   + eps
    power_sum = power.sum() + eps

    # Frequency centre: mean frequency weighted by magnitude (spec definition)
    freq_centre = float(np.sum(freqs * mag) / mag_sum)

    # RMS frequency: sqrt( E[f^2] ) under the power spectrum
    rms_freq = float(np.sqrt(np.sum(freqs ** 2 * power) / power_sum))

    # Frequency variance: E[(f - fc)^2] under the power spectrum
    freq_variance = float(np.sum((freqs - freq_centre) ** 2 * power) / power_sum)

    # Spectral entropy: entropy of the normalised power spectrum
    p_norm        = power / power_sum
    spectral_entropy = float(-np.sum(p_norm * np.log(p_norm + eps)))

    return np.array(
        [freq_centre, rms_freq, freq_variance, spectral_entropy],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Single-window extraction (one file = one timestep)
# ---------------------------------------------------------------------------

def _extract_window(window: np.ndarray, fs: float = FS) -> np.ndarray:
    """Extract 32 features from a single (2560, 2) window.

    Parameters
    ----------
    window : np.ndarray, shape (2560, 2)
    fs     : float — sampling frequency in Hz

    Returns
    -------
    np.ndarray, shape (32,), float32
    """
    if window.ndim != 2 or window.shape[1] != N_CHANNELS:
        raise ValueError(
            f"window must be (samples, {N_CHANNELS}), got {window.shape}"
        )

    parts = []
    for ch in range(N_CHANNELS):          # 0 = H, 1 = V
        sig = window[:, ch].astype(np.float64)
        parts.append(_time_domain_features(sig))
        parts.append(_freq_domain_features(sig, fs=fs))

    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API — primary extraction function
# ---------------------------------------------------------------------------

def extract_features(
    raw_array: np.ndarray,
    fs: float = FS,
    verbose: bool = True,
) -> np.ndarray:
    """Extract a 32-D feature vector for every measurement window.

    Parameters
    ----------
    raw_array : np.ndarray, shape (N_files, 2560, 2)
        Raw accelerometer data as returned by ``data_loader.load_bearing``.
        Axis-2: channel 0 = horizontal, channel 1 = vertical.
    fs : float
        Sampling frequency in Hz (default 25 600 Hz for PRONOSTIA).
    verbose : bool
        Log progress every 10 % of windows.

    Returns
    -------
    np.ndarray, shape (N_files, 32), dtype float32
        Feature matrix; one row per measurement file (timestep).
        Column order follows :data:`FEATURE_NAMES`.
    """
    if raw_array.ndim != 3 or raw_array.shape[2] != N_CHANNELS:
        raise ValueError(
            f"raw_array must be (N_files, samples, {N_CHANNELS}), "
            f"got {raw_array.shape}"
        )

    n_files = raw_array.shape[0]
    features = np.empty((n_files, N_FEATURES), dtype=np.float32)
    log_every = max(1, n_files // 10)

    for i in range(n_files):
        features[i] = _extract_window(raw_array[i], fs=fs)
        if verbose and (i + 1) % log_every == 0:
            logger.info("  extracted %d / %d windows …", i + 1, n_files)

    return features


# ---------------------------------------------------------------------------
# Dataset-level helper (operates on the full bearing dict)
# ---------------------------------------------------------------------------

def extract_dataset_features(
    dataset: Dict[str, Dict[str, np.ndarray]],
    fs: float = FS,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run :func:`extract_features` on every bearing in *dataset*.

    Parameters
    ----------
    dataset : dict
        Output of ``data_loader.load_pronostia``:
        ``{bearing_id: {"data": (N,2560,2), "rul": (N,)}}``.
    fs : float
        Sampling frequency in Hz.
    verbose : bool
        Forward to :func:`extract_features`.

    Returns
    -------
    dict
        ``{bearing_id: {"features": (N, 32), "rul": (N,)}}``
    """
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for bid, arrays in dataset.items():
        logger.info("Extracting features — bearing %s (%d windows) …",
                    bid, arrays["data"].shape[0])
        feat = extract_features(arrays["data"], fs=fs, verbose=verbose)
        result[bid] = {"features": feat, "rul": arrays["rul"]}
        logger.info("  done: features %s", feat.shape)
    return result


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_path: str | Path = "results/scaler.pkl",
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on *X_train*, transform both splits, save scaler.

    The scaler is fitted exclusively on training data to prevent leakage.

    Parameters
    ----------
    X_train : np.ndarray, shape (N_train, 32)
        Training feature matrix (concatenated across training bearings).
    X_test  : np.ndarray, shape (N_test, 32)
        Test feature matrix.
    scaler_path : str or Path
        Where to persist the fitted scaler (``results/scaler.pkl``).

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
        Both feature matrices are float32, shape unchanged.
        ``scaler`` is the fitted :class:`sklearn.preprocessing.StandardScaler`.
    """
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    # Verify fit quality — catch corrupted/wrong-data scaler before saving
    train_means = X_train_scaled.mean(axis=0)
    train_stds  = X_train_scaled.std(axis=0)
    bad_mean = np.where((train_means < -0.01) | (train_means > 0.01))[0]
    bad_std  = np.where((train_stds  < 0.99)  | (train_stds  > 1.01))[0]
    if len(bad_mean) > 0 or len(bad_std) > 0:
        raise ValueError(
            f"Scaler verification failed after fit.\n"
            f"  Features with |mean| > 0.01: {bad_mean.tolist()} "
            f"(values: {train_means[bad_mean].tolist()})\n"
            f"  Features with std outside [0.99,1.01]: {bad_std.tolist()} "
            f"(values: {train_stds[bad_std].tolist()})"
        )
    logger.info(
        "Scaler verification passed: all train means in [-0.01,0.01], stds in [0.99,1.01]"
    )

    joblib.dump(scaler, scaler_path)
    logger.info(
        "Scaler fitted on %d samples and saved to %s", len(X_train), scaler_path
    )

    return X_train_scaled, X_test_scaled, scaler


def load_scaler(scaler_path: str | Path = "results/scaler.pkl") -> StandardScaler:
    """Load a previously saved :class:`StandardScaler`.

    Parameters
    ----------
    scaler_path : str or Path
        Path written by :func:`normalize_features`.

    Returns
    -------
    StandardScaler
    """
    scaler = joblib.load(scaler_path)
    logger.info("Loaded scaler from %s", scaler_path)
    return scaler


# ---------------------------------------------------------------------------
# Save / load processed feature arrays
# ---------------------------------------------------------------------------

def save_features(
    features: Dict[str, Dict[str, np.ndarray]],
    processed_dir: str | Path,
) -> None:
    """Persist extracted features to *processed_dir* as ``.npy`` files.

    Written files: ``{bearing_id}_features.npy``, ``{bearing_id}_rul.npy``.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for bid, arrays in features.items():
        np.save(processed_dir / f"{bid}_features.npy", arrays["features"])
        np.save(processed_dir / f"{bid}_rul.npy",      arrays["rul"])
    logger.info(
        "Saved features for %d bearing(s) to %s.", len(features), processed_dir
    )


def load_features(
    processed_dir: str | Path,
    bearing_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load features previously saved by :func:`save_features`.

    Parameters
    ----------
    processed_dir : str or Path
    bearing_ids   : list[str] or None — subset to load; None loads all.

    Returns
    -------
    dict  ``{bearing_id: {"features": np.ndarray, "rul": np.ndarray}}``
    """
    processed_dir = Path(processed_dir)
    result: Dict[str, Dict[str, np.ndarray]] = {}

    for fp in sorted(processed_dir.glob("*_features.npy")):
        bid = fp.stem.replace("_features", "")
        if bearing_ids is not None and bid not in bearing_ids:
            continue
        rul_fp = processed_dir / f"{bid}_rul.npy"
        if not rul_fp.exists():
            logger.warning("Missing RUL file for bearing %s; skipping.", bid)
            continue
        result[bid] = {
            "features": np.load(fp),
            "rul":      np.load(rul_fp),
        }

    logger.info(
        "Loaded features for %d bearing(s) from %s.", len(result), processed_dir
    )
    return result


# ---------------------------------------------------------------------------
# Health Index computation (uses raw unscaled features)
# ---------------------------------------------------------------------------

def compute_health_index(raw_features_array: np.ndarray) -> np.ndarray:
    """Compute a monotonically-decreasing Health Index from raw (unscaled) features.

    Uses the first 10 % of the bearing's life as a healthy baseline so the
    contrast between healthy and degraded phases is preserved regardless of
    the cross-bearing scaler.

    Parameters
    ----------
    raw_features_array : np.ndarray, shape (N, 32)
        Unscaled feature matrix as returned by :func:`extract_features`.

    Returns
    -------
    np.ndarray, shape (N,), float32
        HI in [0.05, 0.95].  Values near 1 = healthy, near 0 = degraded.
    """
    rms_h  = raw_features_array[:, 0]   # H RMS
    rms_v  = raw_features_array[:, 16]  # V RMS
    kurt_h = raw_features_array[:, 3]   # H Kurtosis
    kurt_v = raw_features_array[:, 19]  # V Kurtosis

    n_baseline = max(10, len(raw_features_array) // 10)

    rms_baseline  = (np.mean(rms_h[:n_baseline])  + np.mean(rms_v[:n_baseline]))  / 2
    kurt_baseline = (np.mean(np.abs(kurt_h[:n_baseline])) +
                     np.mean(np.abs(kurt_v[:n_baseline]))) / 2

    rms_ratio  = (rms_h + rms_v)                         / (2 * rms_baseline  + 1e-8)
    kurt_ratio = (np.abs(kurt_h) + np.abs(kurt_v))       / (2 * kurt_baseline + 1e-8)

    degradation = 0.6 * rms_ratio + 0.4 * kurt_ratio

    # Map to HI: 1 at baseline, decays as degradation grows
    hi_raw = 1.0 / (1.0 + np.log1p(np.maximum(0.0, degradation - 1.0)))

    # Centred rolling mean (window=15)
    hi_smooth = (
        pd.Series(hi_raw)
        .rolling(15, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )

    # Per-bearing min-max normalise to [0.05, 0.95]
    hi_min = float(np.percentile(hi_smooth, 5))
    hi_max = float(np.percentile(hi_smooth, 95))
    hi_norm = (hi_smooth - hi_min) / (hi_max - hi_min + 1e-8)
    hi_norm = np.clip(hi_norm, 0.05, 0.95)

    # Ensure first values > last values (healthy → degraded direction)
    if np.mean(hi_norm[:n_baseline]) < np.mean(hi_norm[-n_baseline:]):
        hi_norm = 1.0 - hi_norm

    return hi_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline precompute (load raw → extract → scale → save)
# ---------------------------------------------------------------------------

def precompute_and_save_features(
    data_dir: str | Path,
    output_dir: str | Path,
    config: Dict[str, Any],
) -> None:
    """Extract features for all PRONOSTIA bearings, fit scaler on train-only,
    save ``{bid}_features.npy``, ``{bid}_rul.npy``, and ``{bid}_hi.npy``.

    Prints ``SCALER VERIFICATION: PASS`` or ``FAIL`` to stdout.
    Raises :exc:`ValueError` if the train-set self-check fails.
    """
    try:
        from src.data_loader import load_pronostia  # type: ignore[import]
    except ImportError:
        from data_loader import load_pronostia  # type: ignore[import]

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support both old 'dataset:' key and new 'data:' key in config
    data_cfg    = config.get("data", config.get("dataset", {}))
    train_bears: List[str] = data_cfg.get("train_bearings", ["1_1", "1_2", "2_1", "2_2", "3_1"])
    test_bears:  List[str] = data_cfg.get("test_bearings",  ["3_2"])
    max_rul: int = int(
        data_cfg.get("rul_clip", config.get("dataset", {}).get("max_rul", 125))
    )
    scaler_path = Path(
        config.get("features", {}).get("scaler_path", "results/scaler.pkl")
    )
    if not scaler_path.is_absolute():
        scaler_path = Path(__file__).resolve().parent.parent / scaler_path

    all_bearings: List[str] = list(train_bears) + [
        b for b in test_bears if b not in train_bears
    ]

    # 1. Load raw data and extract features
    logger.info("Loading %d bearings from %s …", len(all_bearings), data_dir)
    raw_dataset = load_pronostia(data_dir, max_rul=max_rul, bearing_ids=all_bearings)

    feat_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for bid in all_bearings:
        if bid not in raw_dataset:
            logger.warning("Bearing %s not found in %s; skipping.", bid, data_dir)
            continue
        logger.info("  extracting features — bearing %s …", bid)
        feat = extract_features(raw_dataset[bid]["data"], verbose=False)
        feat_dict[bid] = {"features": feat, "rul": raw_dataset[bid]["rul"]}

    # 2. Per-bearing StandardScaler normalization.
    # Each bearing is scaled independently using its own lifetime statistics.
    # This removes cross-condition distribution shift: healthy and degraded
    # phases are in a consistent relative position regardless of operating
    # condition, improving cross-condition RUL generalisation.
    train_present = [b for b in train_bears if b in feat_dict]
    if not train_present:
        raise ValueError("No training bearings found.")
    hi_store: Dict[str, np.ndarray] = {}

    for bid in all_bearings:
        if bid not in feat_dict:
            continue
        X_raw = feat_dict[bid]["features"]   # (N, 32), unscaled
        y_rul = feat_dict[bid]["rul"]

        scaler_b = StandardScaler()
        X_norm   = scaler_b.fit_transform(X_raw).astype(np.float32)
        joblib.dump(scaler_b, output_dir / f"{bid}_scaler.pkl")

        np.save(output_dir / f"{bid}_features.npy", X_norm)
        np.save(output_dir / f"{bid}_rul.npy",      y_rul)

        hi_final = compute_health_index(X_raw)
        np.save(output_dir / f"{bid}_hi.npy", hi_final)
        hi_store[bid] = hi_final

        n_t     = len(hi_final)
        n50     = min(50, max(1, n_t // 4))
        first_m = float(hi_final[:n50].mean())
        last_m  = float(hi_final[-n50:].mean())
        if first_m > 0.70 and last_m < 0.40:
            status = "PASS"
        elif first_m > 0.60 and last_m < 0.50:
            status = "WARN"
        else:
            status = "FAIL"
        logger.log(
            logging.INFO if status == "PASS" else logging.WARNING,
            "  HI check bearing %s: %s (first50=%.3f, last50=%.3f)",
            bid, status, first_m, last_m,
        )
        logger.info(
            "  saved %s: features%s (per-bearing norm)  rul%s  hi%s",
            bid, X_norm.shape, y_rul.shape, hi_final.shape,
        )

    # 4. Plot HI curves for all bearings → results/hi_verification.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, bid in enumerate(all_bearings):
            if bid not in hi_store:
                continue
            hi  = hi_store[bid]
            t   = np.linspace(0, 1, len(hi))
            ax.plot(t, hi, label=f"Bearing {bid}", color=_colors[idx % len(_colors)],
                    linewidth=1.4, alpha=0.85)
        ax.axhline(0.70, color="green",  linestyle="--", linewidth=0.9,
                   label="PASS threshold (first50 > 0.70)")
        ax.axhline(0.40, color="red",    linestyle="--", linewidth=0.9,
                   label="PASS threshold (last50 < 0.40)")
        ax.set_xlabel("Normalised time (0 = new, 1 = failure)")
        ax.set_ylabel("Health Index")
        ax.set_title("PRONOSTIA Health Index — all bearings")
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = Path(__file__).resolve().parent.parent / "results" / "hi_verification.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info("HI verification plot saved to %s", plot_path)
    except Exception as _e:
        logger.warning("Could not save HI plot: %s", _e)

    # 5. Spot-check per-bearing normalization on first training bearing
    bid0   = train_present[0]
    X_chk  = np.load(output_dir / f"{bid0}_features.npy")
    bm_chk = float(np.abs(X_chk.mean()))
    bs_chk = float(X_chk.std())
    if bm_chk < 0.01 and 0.99 < bs_chk < 1.01:
        print("PER-BEARING NORMALIZATION: PASS")
    else:
        print(f"PER-BEARING NORMALIZATION: WARN  ({bid0} |mean|={bm_chk:.4f}, std={bs_chk:.4f})")


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.utils import load_config, set_seed  # type: ignore[import]

    cfg = load_config(_proj / "config.yaml")
    set_seed(cfg.get("seed", 42))

    # Run full pipeline: load raw → extract → scale → save
    precompute_and_save_features(
        _proj / "data" / "raw",
        _proj / "data" / "processed",
        cfg,
    )

    _ok      = True
    proc_dir = _proj / "data" / "processed"

    # 1. Verify 18 expected .npy files (6 bearings × features/rul/hi)
    EXPECTED = [
        f"{bid}_{kind}.npy"
        for bid in ("1_1", "1_2", "2_1", "2_2", "3_1", "3_2")
        for kind in ("features", "rul", "hi")
    ]
    print(f"\nChecking {len(EXPECTED)} expected .npy files in data/processed/:")
    for fname in EXPECTED:
        fp = proc_dir / fname
        if fp.exists():
            shape = np.load(fp, mmap_mode="r").shape
            print(f"  OK      {fname:<30s}  shape={shape}")
        else:
            print(f"  MISSING {fname}  <- ERROR")
            _ok = False

    # 2. Per-bearing normalization check (mean=0, std=1 by construction)
    print("\nPer-bearing normalization check (expect |mean|<0.01, std~1.0):")
    for bid in ("1_1", "1_2", "2_1", "2_2", "3_1", "3_2"):
        fp = proc_dir / f"{bid}_features.npy"
        if fp.exists():
            X      = np.load(fp)
            feat_m = float(np.abs(X.mean()))
            feat_s = float(X.std())
            tag    = "OK  " if feat_m < 0.01 and 0.99 < feat_s < 1.01 else "WARN"
            print(f"  {tag}  {bid}: |mean|={feat_m:.4f}  std={feat_s:.4f}")
            if feat_m > 0.5 or feat_s < 0.5:
                _ok = False

    print()
    if _ok:
        print("FEATURE PIPELINE: READY")
    else:
        print("FEATURE PIPELINE: FAILED")
        sys.exit(1)
