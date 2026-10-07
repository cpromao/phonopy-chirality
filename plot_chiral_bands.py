#!/usr/bin/env python3
"""
Plot a Phonopy phonon band structure (``band.hdf5`` or ``band.yaml``) and
density of states (``mesh.hdf5``), coloured by phonon chirality (angular
momentum).

The phonon angular-momentum component along direction :math:`d` of a
mode with mass-weighted eigenvector :math:`\\xi_{\\alpha,d}` is

.. math::
    \\chi_d \\;=\\; \\sum_\\alpha \\bigl(|\\langle R_d^\\alpha|\\xi\\rangle|^2
                                    - |\\langle L_d^\\alpha|\\xi\\rangle|^2\\bigr)
            \\;=\\; \\sum_\\alpha 2\\,\\mathrm{Im}\\!\\bigl(\\xi^*_{\\alpha,d_1}\\,
                                              \\xi_{\\alpha,d_2}\\bigr)

with :math:`(d_1, d_2)` the cyclic perpendicular pair. The total chirality
:math:`\\|\\boldsymbol\\chi\\|` is bounded in :math:`[0, 1]` in units of
:math:`\\hbar`.

Phonopy output must contain eigenvectors: set ``EIGENVECTORS = .TRUE.``
(``--eigvecs`` on the command line) and, for HDF5 output, ``HDF5 = .TRUE.``
(``--hdf5``); with the Python API, pass ``with_eigenvectors=True`` to
``run_band_structure`` and ``run_mesh``.

Usage
-----
    python plot_chiral_bands.py band.yaml
    python plot_chiral_bands.py band.yaml --component z --unit meV
    python plot_chiral_bands.py band.yaml --ylim 0 50 -o out.png
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.collections import LineCollection

# Optional, only needed for band.hdf5 input
try:
    import h5py  # type: ignore
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

# Detect whether the C-backed YAML parser is available -- without it PyYAML
# falls back to a pure-Python parser that is roughly 10x slower.
try:
    from yaml import CSafeLoader as _YAML_LOADER  # type: ignore
    _HAS_LIBYAML = True
except ImportError:
    from yaml import SafeLoader as _YAML_LOADER   # type: ignore
    _HAS_LIBYAML = False

# Frequency-unit conversion factors from THz
THZ_TO = {
    "THz":  1.0,
    "meV":  4.135667696,
    "cm-1": 33.35641,
}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _extract_yaml(data: dict) -> dict:
    """Pull the arrays we need out of the parsed YAML structure."""
    natom = data["natom"]
    phonons = data["phonon"]
    nqpt = len(phonons)
    nband = len(phonons[0]["band"])

    if "eigenvector" not in phonons[0]["band"][0]:
        raise ValueError(
            "band.yaml has no eigenvectors. Re-run phonopy with "
            "EIGENVECTORS = .TRUE. (--eigvecs), or with_eigenvectors=True "
            "in run_band_structure()."
        )

    distances = np.empty(nqpt)
    frequencies = np.empty((nqpt, nband))                       # THz
    eigvecs = np.empty((nqpt, nband, natom, 3), dtype=complex)  # dyn-mat evecs
    qpoints = np.empty((nqpt, 3))

    for iq, ph in enumerate(phonons):
        distances[iq] = ph["distance"]
        qpoints[iq] = ph["q-position"]
        for ib, b in enumerate(ph["band"]):
            frequencies[iq, ib] = b["frequency"]
            ev = np.asarray(b["eigenvector"])      # (natom, 3, 2)  -> (Re, Im)
            eigvecs[iq, ib] = ev[..., 0] + 1j * ev[..., 1]

    return {
        "natom": natom,
        "nqpt": nqpt,
        "nband": nband,
        "distances": distances,
        "frequencies": frequencies,
        "eigvecs": eigvecs,
        "qpoints": qpoints,
        "segment_nqpoint": data.get("segment_nqpoint", [nqpt]),
        "labels": data.get("labels", None),
    }


def _load_yaml(filename: str | Path) -> dict:
    """Parse a Phonopy ``band.yaml`` file (uses libyaml C loader if available)."""
    if not _HAS_LIBYAML:
        warnings.warn(
            "PyYAML is using the pure-Python parser (libyaml not installed). "
            "Large band.yaml files will load *much* faster after "
            "`pip install --force-reinstall --no-binary :all: pyyaml` "
            "on a system where libyaml-dev is available, or just use "
            "band.hdf5 input instead.",
            stacklevel=2,
        )
    with open(filename, "r") as f:
        return yaml.load(f, Loader=_YAML_LOADER)


def _load_hdf5(filename: str | Path) -> dict:
    """Read a Phonopy ``band.hdf5`` file and return the standard info dict.

    Phonopy stores the dynamical-matrix eigenvector *matrix* at each q-point,
    not per-band rows. The on-disk layout is

        frequency   : (npath, nqpp, nband)                         real
        eigenvector : (npath, nqpp, natom*3, natom*3)              complex
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       inner matrix: rows = atomic Cartesian dof,
                                     COLUMNS = mode index
        distance    : (npath, nqpp)                                real
        path        : (npath, nqpp, 3)                             real
        label       : (npath, 2)  bytes  (optional)

    Modern h5py versions deserialize the complex eigenvector dataset to a
    complex numpy array directly. A few older combinations write a real
    array with a trailing length-2 (Re, Im) axis instead, and we handle
    that as a fallback.

    The function flattens the path dimension so the returned arrays match
    the (nqpt, ...) layout used by the rest of the script.
    """
    if not _HAS_H5PY:
        raise ImportError(
            "h5py is required to read band.hdf5 files. Install with "
            "`pip install h5py`, or use a band.yaml input instead."
        )

    with h5py.File(filename, "r") as f:
        if "eigenvector" not in f:
            raise ValueError(
                "band.hdf5 has no eigenvectors. Re-run phonopy with "
                "EIGENVECTORS = .TRUE. (--eigvecs), or with_eigenvectors=True "
                "in run_band_structure()."
            )

        freq = np.asarray(f["frequency"])
        ev_raw = np.asarray(f["eigenvector"])
        dist = np.asarray(f["distance"])
        path = np.asarray(f["path"])
        # `label` is sometimes a length-0 dataset when no labels were set
        if "label" in f and np.asarray(f["label"]).size > 0:
            label_arr = np.asarray(f["label"])
        else:
            label_arr = None
        if "segment_nqpoint" in f:
            seg_nqpoint = [int(x) for x in np.asarray(f["segment_nqpoint"])]
        else:
            seg_nqpoint = None

    npath, nqpp, nband = freq.shape

    # --- bring eigenvector array to complex form ------------------------- #
    if np.iscomplexobj(ev_raw):
        ev_complex = ev_raw
    elif ev_raw.shape[-1] == 2:
        ev_complex = ev_raw[..., 0] + 1j * ev_raw[..., 1]
    else:
        raise ValueError(
            f"Cannot interpret eigenvector dataset of dtype {ev_raw.dtype} "
            f"and shape {ev_raw.shape}. Expected either a complex array of "
            f"shape (npath, nqpp, natom*3, natom*3), or a real array of "
            f"shape (..., 2) packing (Re, Im) along the last axis."
        )

    # --- check the matrix layout matches our expectation ----------------- #
    if ev_complex.ndim != 4 or ev_complex.shape[2] != ev_complex.shape[3]:
        raise ValueError(
            f"Unexpected eigenvector array shape {ev_complex.shape}. "
            f"Expected (npath, nqpp, natom*3, natom*3)."
        )
    natom_x3 = ev_complex.shape[2]
    if natom_x3 != nband:
        raise ValueError(
            f"Eigenvector matrix size {natom_x3} does not match number of "
            f"bands {nband}; cannot determine natom."
        )
    natom = natom_x3 // 3

    # --- transpose mode axis into front, then split atomic dof into (atom, 3)
    # ev_complex is (npath, nqpp, atomic_dof, mode); swap last two so mode comes first
    ev_complex = np.transpose(ev_complex, (0, 1, 3, 2))
    ev_complex = ev_complex.reshape(npath, nqpp, nband, natom, 3)

    # --- flatten path dimension to the (nqpt, ...) layout ---------------- #
    nqpt = npath * nqpp
    distances = dist.reshape(nqpt)
    frequencies = freq.reshape(nqpt, nband)
    eigvecs = ev_complex.reshape(nqpt, nband, natom, 3)
    qpoints = path.reshape(nqpt, 3)

    if label_arr is not None:
        labels = [[label_arr[i, 0].decode(), label_arr[i, 1].decode()]
                  for i in range(npath)]
    else:
        labels = None

    return {
        "natom": natom,
        "nqpt": nqpt,
        "nband": nband,
        "distances": distances,
        "frequencies": frequencies,
        "eigvecs": eigvecs,
        "qpoints": qpoints,
        "segment_nqpoint": seg_nqpoint if seg_nqpoint is not None
                           else [nqpp] * npath,
        "labels": labels,
    }


def _save_npz_cache(info: dict, cache_path: Path) -> None:
    """Write the parsed arrays to a compressed .npz for fast reload."""
    labels_arr = (np.array(info["labels"], dtype=object)
                  if info["labels"] is not None else None)
    np.savez(
        cache_path,
        natom=info["natom"], nqpt=info["nqpt"], nband=info["nband"],
        distances=info["distances"],
        frequencies=info["frequencies"],
        eigvecs=info["eigvecs"],
        qpoints=info["qpoints"],
        segment_nqpoint=np.asarray(info["segment_nqpoint"]),
        labels=labels_arr if labels_arr is not None else np.array([], dtype=object),
        has_labels=labels_arr is not None,
    )


def _load_npz_cache(cache_path: Path) -> dict:
    """Read a cached .npz produced by ``_save_npz_cache``."""
    with np.load(cache_path, allow_pickle=True) as z:
        labels = z["labels"].tolist() if bool(z["has_labels"]) else None
        return {
            "natom": int(z["natom"]),
            "nqpt": int(z["nqpt"]),
            "nband": int(z["nband"]),
            "distances": z["distances"],
            "frequencies": z["frequencies"],
            "eigvecs": z["eigvecs"],
            "qpoints": z["qpoints"],
            "segment_nqpoint": z["segment_nqpoint"].tolist(),
            "labels": labels,
        }


def load_mesh_file(filename: str | Path, *, verbose: bool = True) -> dict:
    """Read a Phonopy ``mesh.hdf5`` for a proper Brillouin-zone-sampled DOS.

    A band-path DOS is biased: it only samples q-points lying on the chosen
    high-symmetry lines, so it is a *path* density of states, not a true
    BZ-averaged one. For a physically meaningful DOS you need a regular
    q-mesh with symmetry weights, which is what ``mesh.hdf5`` contains.

    Expected datasets (written by ``Phonopy.write_hdf5_mesh()`` when
    ``run_mesh(..., with_eigenvectors=True)`` was used):

        frequency   : (nqpt, nband)                     real
        eigenvector : (nqpt, natom*3, natom*3)          complex
                        rows = atomic dof, COLUMNS = mode
        weight      : (nqpt,)                           int   symmetry weights
        qpoint      : (nqpt, 3)                         real

    Returns a dict with ``frequencies``, ``eigvecs`` (nqpt, nband, natom, 3),
    and ``weights`` (nqpt,).
    """
    if not _HAS_H5PY:
        raise ImportError(
            "h5py is required to read mesh.hdf5. Install with `pip install h5py`."
        )
    src = Path(filename)
    if not src.is_file():
        raise FileNotFoundError(src)

    if verbose:
        print(f"  reading mesh HDF5: {src}")

    with h5py.File(src, "r") as f:
        if "eigenvector" not in f:
            raise ValueError(
                f"{src} has no eigenvectors. Re-run the phonopy mesh with "
                f"EIGENVECTORS = .TRUE. (--eigvecs), or with_eigenvectors=True "
                f"in run_mesh()."
            )
        freq = np.asarray(f["frequency"])
        ev_raw = np.asarray(f["eigenvector"])
        weights = (np.asarray(f["weight"]).astype(float) if "weight" in f
                   else np.ones(freq.shape[0]))
        qpoints = np.asarray(f["qpoint"]) if "qpoint" in f else None

    nqpt, nband = freq.shape

    if np.iscomplexobj(ev_raw):
        ev = ev_raw
    elif ev_raw.shape[-1] == 2:
        ev = ev_raw[..., 0] + 1j * ev_raw[..., 1]
    else:
        raise ValueError(
            f"Cannot interpret mesh eigenvector dataset with dtype "
            f"{ev_raw.dtype} and shape {ev_raw.shape}."
        )

    if ev.ndim != 3 or ev.shape[1] != ev.shape[2]:
        raise ValueError(
            f"Unexpected mesh eigenvector shape {ev.shape}; expected "
            f"(nqpt, natom*3, natom*3)."
        )
    natom = ev.shape[1] // 3
    # columns are modes -> move mode axis in front of the atomic-dof axis
    ev = np.transpose(ev, (0, 2, 1)).reshape(nqpt, nband, natom, 3)

    return {
        "natom": natom,
        "nqpt": nqpt,
        "nband": nband,
        "frequencies": freq,
        "eigvecs": ev,
        "weights": weights,
        "qpoints": qpoints,
    }


def load_band_file(filename: str | Path, *, use_cache: bool = True,
                   verbose: bool = True) -> dict:
    """Load a Phonopy band file (.yaml or .hdf5) into the standard info dict.

    If ``use_cache`` is True and the input is a YAML file, the parsed arrays
    are cached next to the source as ``<filename>.npz``. Subsequent calls
    reuse the cache provided it is newer than the source file. HDF5 inputs
    are already fast enough that no extra cache is written.
    """
    src = Path(filename)
    if not src.is_file():
        raise FileNotFoundError(src)

    suffix = src.suffix.lower()

    # HDF5 path -- already fast, no caching
    if suffix in (".hdf5", ".h5"):
        if verbose:
            print(f"  reading HDF5: {src}")
        return _load_hdf5(src)

    # YAML path -- try cache first
    cache = src.with_suffix(src.suffix + ".npz")
    if use_cache and cache.is_file() and cache.stat().st_mtime >= src.stat().st_mtime:
        if verbose:
            print(f"  reading cache: {cache}")
        try:
            return _load_npz_cache(cache)
        except Exception as exc:  # corrupted / version-mismatched cache
            if verbose:
                print(f"  cache unreadable ({exc}); falling back to YAML")

    if verbose:
        loader = "libyaml" if _HAS_LIBYAML else "pure-Python (slow)"
        print(f"  parsing YAML with {loader}: {src}")
    info = _extract_yaml(_load_yaml(src))

    if use_cache:
        try:
            _save_npz_cache(info, cache)
            if verbose:
                print(f"  wrote cache: {cache}  "
                      f"(next run will skip the YAML parser)")
        except OSError as exc:
            if verbose:
                print(f"  could not write cache ({exc}); continuing")
    return info


# --------------------------------------------------------------------------- #
# Chirality
# --------------------------------------------------------------------------- #
def compute_chirality(eigvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phonon angular momentum for every (q-point, band).

    For each atom :math:`\\alpha` and Cartesian direction :math:`d`:

    .. math::  S_d^\\alpha = 2\\,\\mathrm{Im}(\\xi^*_{\\alpha,d_1}\\xi_{\\alpha,d_2})

    with :math:`(d_1,d_2)` the cyclic perpendicular pair to :math:`d`.

    Parameters
    ----------
    eigvecs : ndarray, shape (nqpt, nband, natom, 3), complex
        Mass-weighted phonon eigenvectors (Phonopy's ``eigenvector`` field).

    Returns
    -------
    chi       : ndarray (nqpt, nband, 3)        components in units of hbar
    chi_total : ndarray (nqpt, nband)           Euclidean norm of chi
    S         : ndarray (nqpt, nband, 3, natom) per-atom contributions
    """
    nqpt, nband, natom, _ = eigvecs.shape
    S = np.empty((nqpt, nband, 3, natom))

    for d in range(3):
        d1 = (d + 1) % 3
        d2 = (d + 2) % 3
        S[:, :, d, :] = 2.0 * np.imag(
            np.conj(eigvecs[..., d1]) * eigvecs[..., d2]
        )

    chi = S.sum(axis=-1)
    chi_total = np.linalg.norm(chi, axis=-1)
    return chi, chi_total, S


def zero_degenerate(
    frequencies: np.ndarray,
    chi: np.ndarray,
    chi_total: np.ndarray,
    S: np.ndarray,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set the chirality of (near-)degenerate modes to zero.

    Within a degenerate subspace the individual eigenvectors are not unique
    (any unitary mixing is also an eigenvector), so the per-mode angular
    momentum is gauge-dependent and not physically meaningful. Following the
    original MATLAB script we suppress those values.

    ``tol`` is a *relative* tolerance, with an absolute floor of 1 (so that
    near-zero acoustic frequencies at Gamma aren't all flagged as degenerate
    with each other through round-off only).
    """
    chi = chi.copy()
    chi_total = chi_total.copy()
    S = S.copy()
    nqpt, nband = frequencies.shape

    for iq in range(nqpt):
        for ib in range(nband):
            f = frequencies[iq, ib]
            scale = max(abs(f), 1.0)
            is_degen = False
            if ib > 0 and abs(f - frequencies[iq, ib - 1]) < tol * scale:
                is_degen = True
            if ib + 1 < nband and abs(f - frequencies[iq, ib + 1]) < tol * scale:
                is_degen = True
            if is_degen:
                chi[iq, ib, :] = 0.0
                chi_total[iq, ib] = 0.0
                S[iq, ib, :, :] = 0.0
    return chi, chi_total, S


def sort_modes_by_frequency(
    frequencies: np.ndarray,
    chi: np.ndarray,
    chi_total: np.ndarray,
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reorder bands so that at each q-point band index = frequency rank.

    Phonopy's ``is_band_connection=True`` tries to track band identity across
    q-points via eigenvector overlap, but the algorithm can fail badly in
    crowded spectra (lots of crossings, near-degeneracies), producing the
    classic "wild zig-zag" artifact. Sorting modes by frequency at each
    q-point eliminates the artifact at the cost of band identity: each
    plotted line is now the n-th lowest frequency, not a continuously
    tracked mode.

    Chirality data is reordered with the same permutation so the colored
    value at each (q, frequency) point still belongs to the mode that
    actually lives there.
    """
    order = np.argsort(frequencies, axis=1)
    f_sorted = np.take_along_axis(frequencies, order, axis=1)
    chit_sorted = np.take_along_axis(chi_total, order, axis=1)
    chi_sorted = np.take_along_axis(chi, order[..., None], axis=1)
    # S has shape (nqpt, nband, 3, natom): use broadcasting on the band axis
    S_sorted = np.take_along_axis(S, order[..., None, None], axis=1)
    return f_sorted, chi_sorted, chit_sorted, S_sorted


def compute_chirality_histogram(
    frequencies: np.ndarray,
    color_values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    nbins: int = 150,
    binwidth: float | None = None,
    frange: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Phonon DOS as a histogram, with the mean of a per-mode value in each bin.

    Each mode contributes its q-point weight :math:`w_{\mathbf q}` (normalized
    so that :math:`\sum_{\mathbf q} w_{\mathbf q} = 1`) to the bin containing
    its frequency, giving

    .. math::
        g_b = \frac{1}{\Delta\omega_b} \sum_{\mathbf q\nu \in b} w_{\mathbf q},
        \qquad
        \langle X \rangle_b = \frac{\sum_{\mathbf q\nu \in b} w_{\mathbf q}\, X_{\mathbf q\nu}}
                                   {\sum_{\mathbf q\nu \in b} w_{\mathbf q}},

    so :math:`\sum_b g_b \Delta\omega_b` is the number of modes per cell in
    the binned range, and :math:`\langle X\rangle_b` averages over exactly the
    modes that make up bar :math:`b`.

    Parameters
    ----------
    frequencies  : (nqpt, nband)  mode frequencies
    color_values : (nqpt, nband)  per-mode quantity to average (e.g. |J|)
    weights      : (nqpt,) or None   q-point symmetry weights (None -> uniform)
    nbins        : number of bins across ``frange`` (ignored if ``binwidth``)
    binwidth     : bin width in the unit of ``frequencies``
    frange       : (fmin, fmax) to bin; default is the full frequency range

    Returns
    -------
    edges : (nb + 1,)  bin edges
    dos   : (nb,)      density of states, states per unit frequency per cell
    mean  : (nb,)      weighted mean of ``color_values`` (NaN for empty bins)
    """
    freqs = np.asarray(frequencies, dtype=float)
    vals = np.asarray(color_values, dtype=float)
    nqpt = freqs.shape[0]

    w = np.ones(nqpt) if weights is None else np.asarray(weights, dtype=float)
    w = np.broadcast_to((w / w.sum())[:, None], freqs.shape)

    lo, hi = (float(freqs.min()), float(freqs.max())) if frange is None \
        else (float(min(frange)), float(max(frange)))
    if hi <= lo:
        hi = lo + 1.0
    if binwidth is not None and binwidth > 0:
        n = max(1, int(np.ceil((hi - lo) / binwidth - 1e-9)))
        edges = lo + binwidth * np.arange(n + 1)
    else:
        edges = np.linspace(lo, hi, max(1, int(nbins)) + 1)

    counts, _ = np.histogram(freqs.ravel(), bins=edges, weights=w.ravel())
    acc, _ = np.histogram(freqs.ravel(), bins=edges, weights=(w * vals).ravel())
    dos = counts / np.diff(edges)
    mean = np.full(counts.shape, np.nan)
    filled = counts > 0
    mean[filled] = acc[filled] / counts[filled]
    return edges, dos, mean


def _draw_chirality_histogram(ax, edges, dos, mean, cmap, norm, *, horizontal):
    """Draw the DOS histogram with each bar filled by its mean chirality.

    A thin dark outline traces the DOS envelope, so bins whose colour is
    close to the background (e.g. J ~ 0 with a diverging colormap) stay
    visible.
    """
    cmap = plt.get_cmap(cmap)
    t = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(norm(np.nan_to_num(mean)),
                                                         dtype=float)), 0.0)
    colours = cmap(np.clip(t, 0.0, 1.0))
    keep = dos > 0
    lo, width = edges[:-1][keep], np.diff(edges)[keep]
    style = dict(align="edge", color=colours[keep], edgecolor=colours[keep],
                 linewidth=0.3, zorder=2)
    if horizontal:
        ax.barh(lo, dos[keep], height=width, **style)
    else:
        ax.bar(lo, dos[keep], width=width, **style)
    ax.stairs(dos, edges, orientation="horizontal" if horizontal else "vertical",
              color="0.25", linewidth=0.6, zorder=3)


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
# Seekpath / SeeK-path uppercase names → LaTeX command strings.
# Ordered longest-first so "LAMBDA" matches before "LA" etc.
_SEEKPATH_GREEK = [
    ("GAMMA",  r"\Gamma"),
    ("LAMBDA", r"\Lambda"),
    ("SIGMA",  r"\Sigma"),
    ("DELTA",  r"\Delta"),
    ("THETA",  r"\Theta"),
    ("OMEGA",  r"\Omega"),
    ("LAMBDA", r"\Lambda"),
    ("PHI",    r"\Phi"),
    ("PSI",    r"\Psi"),
    ("XI",     r"\Xi"),
    ("PI",     r"\Pi"),
]


def _fmt_single(s: str) -> str:
    """Return the inner LaTeX string for one label token (no pipes).

    Examples::
        "GAMMA"    → "\\Gamma"
        "$\\Gamma$" → "\\Gamma"         (math-wrapped, as in Phonopy BAND_LABELS)
        "Γ"        → "\\Gamma"         (Unicode)
        "SIGMA_0"  → "\\Sigma_0"
        "C_0"      → "C_0"              (stays as-is; underscore triggers math)
        "Z"        → "Z"
    """
    s = s.strip()
    # Strip an existing $...$ wrapper; the caller adds a single one back
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if not s:
        return ""
    # Already LaTeX inner content
    if s.startswith("\\"):
        return s
    if s == "Γ":
        return r"\Gamma"
    su = s.upper()
    for name, latex in _SEEKPATH_GREEK:
        if su == name:
            return latex
        # Match e.g. SIGMA_0, SIGMA0, SIGMA1
        if su.startswith(name) and len(su) > len(name) and su[len(name)] in "_0123456789":
            suffix = s[len(name):]   # preserve original capitalisation of suffix
            return latex + suffix
    return s


def _format_label(label: str | None) -> str:
    """Render a high-symmetry-point label (possibly pipe-joined) for matplotlib.

    Handles seekpath conventions (SIGMA_0, LAMBDA, …), math-wrapped labels
    (``$\\Gamma$``) and a Unicode Γ.  Pipe-joined discontinuity labels
    (e.g. ``"C_0|SIGMA_0"``) are emitted as a single math expression
    ``$C_0|\\Sigma_0$`` so the vertical bar renders cleanly.
    """
    if label is None or label == "":
        return ""
    parts = [p.strip() for p in label.strip().split("|")]
    inner = [_fmt_single(p) for p in parts]
    # Use math mode when any part contains LaTeX commands or sub/superscripts
    needs_math = any(c in p for p in inner for c in ("\\", "_", "^"))
    joined = "|".join(inner)
    return f"${joined}$" if needs_math else joined


def _tick_info(distances, segment_nqpoint, labels):
    """Build (positions, labels) for the band-path x-axis."""
    nseg = len(segment_nqpoint)
    seg_starts, idx = [], 0
    for n in segment_nqpoint:
        seg_starts.append(idx)
        idx += n
    seg_ends = [seg_starts[i] + segment_nqpoint[i] - 1 for i in range(nseg)]

    if labels is None:
        positions = [distances[seg_starts[0]]] + [distances[e] for e in seg_ends]
        return positions, [""] * len(positions)

    positions, ticks = [distances[seg_starts[0]]], [_format_label(labels[0][0])]
    for i in range(nseg - 1):
        end_lbl  = labels[i][1]
        next_lbl = labels[i + 1][0]
        positions.append(distances[seg_ends[i]])
        # Combine into one token so pipe-joined labels get a single $...$ block
        combined = end_lbl if end_lbl == next_lbl else f"{end_lbl}|{next_lbl}"
        ticks.append(_format_label(combined))
    positions.append(distances[seg_ends[-1]])
    ticks.append(_format_label(labels[-1][1]))
    return positions, ticks


# Angular momenta below this (in units of hbar) are numerical noise; a data-
# driven colour limit smaller than this would stretch noise across the scale.
_J_NOISE_FLOOR = 1e-6


def _colour_limits(absv, args, signed):
    """Resolve (vmin, vmax) from --vmin/--vmax/--clim-pct and the data.

    Explicit --vmin/--vmax always win. Data-driven limits (--clim-pct, and
    the signed-component default of max|J|) fall back to 1 when the data are
    effectively zero, as for achiral or centrosymmetric crystals; otherwise
    rounding noise would be rendered as if it were genuine chirality.
    """
    auto = None
    if args.clim_pct is not None:
        auto = float(np.percentile(absv, args.clim_pct)) if absv.size else 0.0
        if auto < _J_NOISE_FLOOR:
            auto = float(np.max(absv)) if absv.size else 0.0
    elif signed:
        auto = float(np.max(absv)) if absv.size else 0.0
    if auto is not None and auto < _J_NOISE_FLOOR:
        if args.vmax is None:
            print(f"  ↳ chirality is below {_J_NOISE_FLOOR:g} hbar everywhere "
                  f"(numerically zero); using the full-scale colour limit")
        auto = 1.0

    if auto is None:                          # |J| default: physical bound
        vmax = args.vmax if args.vmax is not None else 1.0
        vmin = args.vmin if args.vmin is not None else 0.0
    else:
        vmax = args.vmax if args.vmax is not None else auto
        vmin = args.vmin if args.vmin is not None else (-abs(vmax) if signed else 0.0)
    return vmin, vmax


def make_norm(vmin, vmax, *, scale="linear", gamma=0.5, signed=False,
              linthresh_frac=0.02):
    """Build a matplotlib colour norm.

    Phonon angular momentum is usually heavily concentrated near zero with a
    thin high-|J| tail, so a linear scale spends nearly the whole colormap on
    modes that are all effectively achiral. ``power`` (gamma < 1) or ``log``
    scaling expands the low end and makes the chiral minority visible.

    scale
        ``linear``  standard.
        ``power``   ``PowerNorm(gamma)``; gamma<1 stretches low values.
                    For signed data the gamma is applied to |v| symmetrically.
        ``log``     ``LogNorm`` (unsigned) or ``SymLogNorm`` (signed).
    """
    import matplotlib.colors as mcolors

    if scale == "linear":
        return plt.Normalize(vmin, vmax)

    if scale == "power":
        if signed:
            # Symmetric power scaling around zero
            lim = max(abs(vmin), abs(vmax))
            return mcolors.SymLogNorm(
                linthresh=max(lim * linthresh_frac, 1e-12),
                vmin=-lim, vmax=lim,
            ) if gamma is None else _SymPowerNorm(gamma, -lim, lim)
        lo = max(vmin, 0.0)
        return mcolors.PowerNorm(gamma, vmin=lo, vmax=vmax)

    if scale == "log":
        if signed:
            lim = max(abs(vmin), abs(vmax))
            return mcolors.SymLogNorm(
                linthresh=max(lim * linthresh_frac, 1e-12),
                vmin=-lim, vmax=lim,
            )
        # LogNorm cannot start at 0; use a small positive floor
        lo = vmin if vmin > 0 else max(vmax * 1e-4, 1e-12)
        return mcolors.LogNorm(vmin=lo, vmax=vmax)

    raise ValueError(f"unknown colour scale {scale!r}")


class _SymPowerNorm(plt.Normalize):
    """Power-law normalisation that is symmetric about zero.

    Maps v -> 0.5 * (1 + sign(v) * |v/lim|**gamma), so gamma < 1 expands the
    region near zero while keeping the diverging colormap centred at 0.5.
    """

    def __init__(self, gamma, vmin, vmax):
        super().__init__(vmin, vmax)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        v = np.ma.asarray(value, dtype=float)
        lim = max(abs(self.vmin), abs(self.vmax)) or 1.0
        t = np.clip(v / lim, -1.0, 1.0)
        out = 0.5 * (1.0 + np.sign(t) * np.abs(t) ** self.gamma)
        return np.ma.masked_array(out, mask=np.ma.getmask(v))


def plot_band_chirality(
    distances, frequencies, color_values,
    segment_nqpoint, labels,
    *,
    freq_unit="THz",
    cmap="viridis",
    vmin=None, vmax=None,
    colorbar_label=r"$\|\mathbf{J}\|\ [\hbar]$",
    output="band_chirality.png",
    ylim=None,
    linewidth=1.5,
    figsize=(10, 6),
    dpi=300,
    fontsize=16,
    dos=None,
    dos_label="DOS",
    norm=None,
):
    """Render the band structure, coloring each band by ``color_values``.

    If ``dos`` is given it must be an ``(edges, dos, mean)`` tuple as
    returned by :func:`compute_chirality_histogram`; it is drawn as a
    histogram side panel sharing the frequency axis, each bar filled with the
    mean angular momentum of the modes in it.
    """
    nqpt, nband = frequencies.shape

    if norm is None:
        if vmin is None:
            vmin = float(np.min(color_values))
        if vmax is None:
            vmax = float(np.max(color_values))
        norm = plt.Normalize(vmin, vmax)

    if dos is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax_dos = None
    else:
        fig, (ax, ax_dos) = plt.subplots(
            1, 2, figsize=(figsize[0] * 1.28, figsize[1]),
            sharey=True,
            gridspec_kw={"width_ratios": [3.2, 1.0], "wspace": 0.04},
        )

    # Segment starts so we don't draw across path discontinuities
    seg_starts, idx = [], 0
    for n in segment_nqpoint:
        seg_starts.append(idx)
        idx += n

    for seg_i, n in enumerate(segment_nqpoint):
        start = seg_starts[seg_i]
        end = start + n
        x_seg = distances[start:end]
        if x_seg.size < 2:
            continue
        for ib in range(nband):
            y_seg = frequencies[start:end, ib]
            c_seg = color_values[start:end, ib]
            pts = np.column_stack([x_seg, y_seg]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_c = 0.5 * (c_seg[:-1] + c_seg[1:])
            lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=linewidth)
            lc.set_array(seg_c)
            ax.add_collection(lc)

    # X axis
    tick_pos, tick_lbl = _tick_info(distances, segment_nqpoint, labels)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=fontsize)
    for p in tick_pos:
        ax.axvline(p, color="k", linewidth=0.5, alpha=0.4)
    ax.set_xlim(distances.min(), distances.max())

    # Y axis
    if ylim is None:
        fmin, fmax = float(np.min(frequencies)), float(np.max(frequencies))
        pad = 0.05 * (fmax - fmin if fmax > fmin else 1.0)
        ax.set_ylim(fmin - pad, fmax + pad)
    else:
        ax.set_ylim(ylim)
    ax.set_ylabel(f"Frequency [{freq_unit}]", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize - 2)
    ax.axhline(0, color="k", linewidth=0.5)

    # DOS side panel, coloured by the DOS-weighted mean angular momentum
    if ax_dos is not None:
        edges, dos_vals, dos_col = dos
        _draw_chirality_histogram(ax_dos, edges, dos_vals, dos_col, cmap, norm,
                                  horizontal=True)

        ax_dos.set_xlim(0, float(np.max(dos_vals)) * 1.08 or 1.0)
        ax_dos.set_xlabel(dos_label, fontsize=fontsize)
        ax_dos.tick_params(axis="x", labelsize=fontsize - 4)
        ax_dos.tick_params(axis="y", left=False)
        ax_dos.axhline(0, color="k", linewidth=0.5)
        ax_dos.grid(axis="x", alpha=0.25)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = ax if ax_dos is None else [ax, ax_dos]
    cbar = plt.colorbar(sm, ax=cbar_ax)
    cbar.set_label(colorbar_label, rotation=270, labelpad=24, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    if ax_dos is None:
        fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {output}")
    return fig, ax


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _warn_signed_dos(args):
    """Signed J components carry no information in a DOS of a non-magnetic crystal."""
    if args.component != "total":
        print(f"  ⚠️ J_{args.component} is odd under time reversal, J(-q) = -J(q), "
              f"so its DOS average vanishes over the full zone (and on a "
              f"symmetry-reduced mesh depends on which of ±q is stored); the "
              f"DOS is only meaningful for --component total")


def _write_dos_csv(path, edges, dos_vals, mean, args, source):
    """Write the DOS histogram: bin edges, DOS and mean chirality per bin."""
    comp = "J" if args.component == "total" else "J" + args.component
    hdr = (f"bin_lo[{args.unit}],bin_hi[{args.unit}],"
           f"dos[states/{args.unit}/cell],mean_{comp}[hbar]  "
           f"({source}-derived; mean is nan for empty bins)")
    np.savetxt(path, np.column_stack([edges[:-1], edges[1:], dos_vals, mean]),
               delimiter=",", header=hdr, comments="# ")
    print(f"Saved DOS data to {path}")


def _dos_only_main(args):
    """Produce a standalone chirality-coloured DOS figure from a q-mesh."""
    mesh = load_mesh_file(args.mesh)
    print(f"  mesh: nqpt = {mesh['nqpt']}, nband = {mesh['nband']}")

    chi, chit, S = compute_chirality(mesh["eigvecs"])
    if not args.no_degen_zero:
        chi, chit, S = zero_degenerate(mesh["frequencies"], chi, chit, S,
                                       tol=args.degen_tol)

    if args.component == "total":
        color = chit
        cb_label = r"$\|\mathbf{J}\|\ [\hbar]$"
        default_cmap = "viridis"
    else:
        color = chi[..., {"x": 0, "y": 1, "z": 2}[args.component]]
        cb_label = rf"$J_{args.component}\ [\hbar]$"
        default_cmap = "bwr"
    cmap = args.cmap if args.cmap is not None else default_cmap
    signed = args.component != "total"

    freqs = mesh["frequencies"] * THZ_TO[args.unit]

    absv = np.abs(color)
    pcts = {p: float(np.percentile(absv, p)) for p in (50, 90, 99, 100)}
    print(f"  |value| percentiles: 50%={pcts[50]:.4f}  90%={pcts[90]:.4f}  "
          f"99%={pcts[99]:.4f}  max={pcts[100]:.4f}")

    vmin, vmax = _colour_limits(absv, args, signed)

    norm = make_norm(vmin, vmax, scale=args.cscale, gamma=args.cgamma,
                     signed=signed)
    print(f"  colour range: [{vmin:.4g}, {vmax:.4g}]  scale={args.cscale}")

    _warn_signed_dos(args)
    edges, dos_vals, dos_col = compute_chirality_histogram(
        freqs, color, mesh["weights"], nbins=args.dos_bins,
        binwidth=args.dos_binwidth, frange=args.ylim,
    )
    if args.dos_out:
        _write_dos_csv(args.dos_out, edges, dos_vals, dos_col, args, "mesh")

    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_chirality_histogram(ax, edges, dos_vals, dos_col, cmap, norm,
                              horizontal=False)

    ax.set_xlim(args.ylim if args.ylim else (edges[0], edges[-1]))
    ax.set_ylim(0, float(np.max(dos_vals)) * 1.08 or 1.0)
    ax.set_xlabel(f"Frequency [{args.unit}]", fontsize=args.fontsize)
    ax.set_ylabel(f"DOS [states/{args.unit}/cell]", fontsize=args.fontsize)
    ax.tick_params(labelsize=args.fontsize - 2)
    ax.axvline(0, color="k", linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cb_label, rotation=270, labelpad=24, fontsize=args.fontsize)
    cbar.ax.tick_params(labelsize=args.fontsize - 2)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Plot phonon band structure colored by phonon chirality "
                    "from a Phonopy band.yaml file.",
    )
    p.add_argument("band_yaml", nargs="?", default="band.yaml",
                   help="Path to band.yaml or band.hdf5 (default: ./band.yaml)")
    p.add_argument("-o", "--output", default="band_chirality.png",
                   help="Output figure file")
    p.add_argument("--component", choices=["total", "x", "y", "z"],
                   default="total",
                   help="Chirality component to color by (default: total norm)")
    p.add_argument("--unit", choices=list(THZ_TO), default="THz",
                   help="Frequency unit for the plot")
    p.add_argument("--cmap", default=None,
                   help="Matplotlib colormap "
                        "(default: 'viridis' for total, 'bwr' for x/y/z)")
    p.add_argument("--vmin", type=float, default=None,
                   help="Lower colour limit (default: 0 for |J|, "
                        "-max|J| for signed components).")
    p.add_argument("--vmax", type=float, default=None,
                   help="Upper colour limit (default: 1 for |J|, "
                        "+max|J| for signed components). Lowering this is "
                        "the simplest way to bring out weak chirality.")
    p.add_argument("--clim-pct", type=float, default=None, metavar="P",
                   help="Set the upper colour limit to the P-th percentile "
                        "of the chirality data instead of a fixed value "
                        "(e.g. --clim-pct 99). Useful when a few outlier "
                        "modes would otherwise compress the whole scale.")
    p.add_argument("--cscale", choices=["linear", "power", "log"],
                   default="linear",
                   help="Colour scale mapping. Phonon |J| is usually sharply "
                        "peaked near zero, so 'power' (with --cgamma < 1) or "
                        "'log' often reveals far more structure than the "
                        "default linear scale.")
    p.add_argument("--cgamma", type=float, default=0.5,
                   help="Exponent for --cscale power (default 0.5). Values "
                        "below 1 expand the low-|J| end.")
    p.add_argument("--ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"),
                   default=None)
    p.add_argument("--labels", default=None,
                   help="Override or supply x-axis labels for the high-"
                        "symmetry points. Provide one label per breakpoint, "
                        "separated by spaces or commas. For N segments give "
                        "N+1 labels; use a pipe inside a label "
                        "(e.g. \"X|Y\") to mark a path discontinuity. "
                        "Example: --labels \"G X M G R\"")
    p.add_argument("--no-sort-freq", action="store_true",
                   help="Keep phonopy band-connection ordering instead of "
                        "sorting modes by frequency. The default is to sort, "
                        "which eliminates zig-zag artifacts from failed band "
                        "connection; disable only if you need true band "
                        "identity along the path.")
    p.add_argument("--fontsize", type=int, default=16,
                   help="Base font size for axis labels and ticks "
                        "(default: 16).")
    p.add_argument("--no-degen-zero", action="store_true",
                   help="Do not zero out chirality on (near-)degenerate modes")
    p.add_argument("--degen-tol", type=float, default=1e-5,
                   help="Relative tolerance for degeneracy detection")
    p.add_argument("--linewidth", type=float, default=1.5)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-cache", action="store_true",
                   help="Don't read or write the .npz cache next to a "
                        "band.yaml input (HDF5 input is never cached).")
    p.add_argument("--dos", action="store_true",
                   help="Add a DOS side panel, coloured by the DOS-weighted "
                        "mean angular momentum at each frequency.")
    p.add_argument("--mesh", default=None, metavar="MESH_HDF5",
                   help="mesh.hdf5 (with eigenvectors) to build a proper "
                        "BZ-sampled DOS. Strongly recommended with --dos: "
                        "without it the DOS is built from the band-path "
                        "q-points only, which samples high-symmetry lines "
                        "rather than the whole zone.")
    p.add_argument("--dos-bins", type=int, default=150,
                   help="Number of DOS histogram bins across the frequency "
                        "range (the --ylim window if given). Default 150.")
    p.add_argument("--dos-binwidth", type=float, default=None,
                   help="DOS histogram bin width in the plot's frequency "
                        "unit; overrides --dos-bins.")
    p.add_argument("--dos-only", action="store_true",
                   help="Plot only the chirality-coloured DOS (no band "
                        "panel). Implied when a mesh.hdf5 is given as the "
                        "input file.")
    p.add_argument("--dos-out", default=None, metavar="CSV",
                   help="Write the DOS data (frequency, DOS, mean |J|) to a "
                        "CSV file.")
    args = p.parse_args(argv)

    if not Path(args.band_yaml).is_file():
        print(f"Error: file '{args.band_yaml}' not found.", file=sys.stderr)
        sys.exit(1)

    # Allow a mesh.hdf5 to be passed as the sole input -> DOS-only figure
    _in = Path(args.band_yaml)
    if args.mesh is None and _in.name.startswith("mesh") and _in.suffix in (".hdf5", ".h5"):
        print(f"Input looks like a q-mesh; producing a DOS-only figure.")
        args.mesh = str(_in)
        args.dos = True
        args.dos_only = True

    if args.dos_only:
        if args.mesh is None:
            print("Error: --dos-only requires --mesh (or a mesh.hdf5 input).",
                  file=sys.stderr)
            sys.exit(1)
        return _dos_only_main(args)

    print(f"Loading {args.band_yaml} ...")
    info = load_band_file(args.band_yaml, use_cache=not args.no_cache)

    # If --dos was asked for without --mesh, look for mesh.hdf5 alongside
    if args.dos and args.mesh is None:
        _cand = Path(args.band_yaml).parent / "mesh.hdf5"
        if _cand.is_file():
            print(f"  found {_cand}; using it for a BZ-sampled DOS")
            args.mesh = str(_cand)
    print(f"  natom = {info['natom']},  nqpt = {info['nqpt']},  "
          f"nband = {info['nband']}")

    # User-supplied labels override whatever was in the file
    if args.labels is not None:
        import re
        user_labels = re.split(r"[\s,]+", args.labels.strip())
        user_labels = [lbl for lbl in user_labels if lbl]
        npath = len(info["segment_nqpoint"])
        if len(user_labels) != npath + 1:
            print(f"Error: --labels has {len(user_labels)} entries but the "
                  f"path has {npath} segments (needs {npath + 1} labels).",
                  file=sys.stderr)
            sys.exit(1)
        info["labels"] = [[user_labels[i], user_labels[i + 1]]
                          for i in range(npath)]

    print("Computing chirality ...")
    chi, chi_total, S = compute_chirality(info["eigvecs"])
    if not args.no_sort_freq:
        print("  sorting modes by frequency at each q-point")
        info["frequencies"], chi, chi_total, S = sort_modes_by_frequency(
            info["frequencies"], chi, chi_total, S,
        )
    if not args.no_degen_zero:
        chi, chi_total, S = zero_degenerate(
            info["frequencies"], chi, chi_total, S, tol=args.degen_tol,
        )

    # Pick component to color by
    if args.component == "total":
        color = chi_total
        cb_label = r"$\|\mathbf{J}\|\ [\hbar]$"
        default_cmap = "viridis"
    else:
        idx = {"x": 0, "y": 1, "z": 2}[args.component]
        color = chi[..., idx]
        cb_label = rf"$J_{args.component}\ [\hbar]$"
        default_cmap = "bwr"

    # Frequency unit conversion
    freqs = info["frequencies"] * THZ_TO[args.unit]

    cmap = args.cmap if args.cmap is not None else default_cmap

    # ------------------------------------------------- colour range report
    _absv = np.abs(color)
    _pcts = {p: float(np.percentile(_absv, p)) for p in (50, 90, 99, 100)}
    print(f"  |value| percentiles: "
          f"50%={_pcts[50]:.4f}  90%={_pcts[90]:.4f}  "
          f"99%={_pcts[99]:.4f}  max={_pcts[100]:.4f}")
    if _pcts[100] >= _J_NOISE_FLOOR and _pcts[99] < 0.25 * _pcts[100]:
        print("  ↳ distribution is strongly peaked near zero; consider "
              "--clim-pct 99, a smaller --vmax, or --cscale power")

    signed = args.component != "total"

    # Default colour range, applied independently so --vmax alone works:
    #   - signed components -> symmetric about 0
    #   - total norm        -> [0, 1] (|J| <= 1 for a normalized mode)
    vmin, vmax = _colour_limits(_absv, args, signed)

    colour_norm = make_norm(vmin, vmax, scale=args.cscale,
                            gamma=args.cgamma, signed=signed)
    print(f"  colour range: [{vmin:.4g}, {vmax:.4g}]  scale={args.cscale}"
          + (f" (gamma={args.cgamma})" if args.cscale == "power" else ""))

    # ---------------------------------------------------------------- DOS
    dos_tuple = None
    if args.dos or args.mesh or args.dos_out:
        if args.mesh:
            mesh = load_mesh_file(args.mesh)
            print(f"  mesh: nqpt = {mesh['nqpt']}, nband = {mesh['nband']}")
            m_chi, m_chit, m_S = compute_chirality(mesh["eigvecs"])
            if not args.no_degen_zero:
                m_chi, m_chit, m_S = zero_degenerate(
                    mesh["frequencies"], m_chi, m_chit, m_S,
                    tol=args.degen_tol,
                )
            if args.component == "total":
                m_color = m_chit
            else:
                m_color = m_chi[..., {"x": 0, "y": 1, "z": 2}[args.component]]
            dos_freqs = mesh["frequencies"] * THZ_TO[args.unit]
            dos_weights = mesh["weights"]
            dos_source = "mesh"
        else:
            print("  ⚠️ building DOS from band-path q-points (no --mesh "
                  "given): this samples high-symmetry lines, not the whole "
                  "Brillouin zone, so it is a path DOS rather than a true one.")
            dos_freqs = freqs
            m_color = color
            dos_weights = None
            dos_source = "path"

        _warn_signed_dos(args)
        edges, dos_vals, dos_col = compute_chirality_histogram(
            dos_freqs, m_color, dos_weights, nbins=args.dos_bins,
            binwidth=args.dos_binwidth, frange=args.ylim,
        )
        dos_tuple = (edges, dos_vals, dos_col)
        if args.dos_out:
            _write_dos_csv(args.dos_out, edges, dos_vals, dos_col, args,
                           dos_source)

    plot_band_chirality(
        info["distances"], freqs, color,
        info["segment_nqpoint"], info["labels"],
        freq_unit=args.unit, cmap=cmap, vmin=vmin, vmax=vmax,
        norm=colour_norm,
        colorbar_label=cb_label, output=args.output,
        ylim=args.ylim, linewidth=args.linewidth, dpi=args.dpi,
        fontsize=args.fontsize,
        dos=dos_tuple,
        dos_label=("DOS" if dos_tuple is None or dos_source == "mesh"
                   else "Path DOS"),
    )


if __name__ == "__main__":
    main()
