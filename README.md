# phonopy-chirality

Post-processing for [Phonopy](https://phonopy.github.io/phonopy/) that resolves the angular momentum of every phonon mode and colours band structures and densities of states (DOS) by it; circularly polarized branches then stand out against the achiral background.

`plot_chiral_bands.py` reads Phonopy's `band.hdf5` (or `band.yaml`) and `mesh.hdf5`, computes the angular momentum of each mode from its eigenvector, and plots the band structure, a chirality-resolved DOS, or both. Phonopy needs to be run with eigenvectors enabled; the settings, and the changes needed in scripts that call Phonopy through its Python API, are described below.

## Phonon angular momentum

For a mode with mass-weighted eigenvector $\xi_{\alpha d}$ (atom $\alpha$, Cartesian direction $d$), the angular momentum along $d$ is the difference between its projections onto right- and left-circularly polarized basis vectors in the plane perpendicular to $d$,

$$
J_d = \sum_\alpha \left( |\langle R^\alpha_d|\xi\rangle|^2 - |\langle L^\alpha_d|\xi\rangle|^2 \right)
    = \sum_\alpha 2\,\mathrm{Im}\left(\xi^*_{\alpha d_1}\,\xi_{\alpha d_2}\right),
$$

where $(d, d_1, d_2)$ is a cyclic permutation of $(x, y, z)$. In units of $\hbar$, $\|\mathbf{J}\| \le 1$ for any normalized mode, with the bound reached when every atom revolves coherently in a single plane. Within a degenerate subspace the individual eigenvectors are fixed only up to a unitary rotation, which makes the per-mode angular momentum gauge-dependent; by default these modes are therefore set to zero (`--no-degen-zero` disables this).

The DOS is drawn as a histogram whose bars are filled with the mean angular momentum of the modes they contain. For bin $b$ of width $\Delta\omega_b$,

$$
g_b = \frac{1}{\Delta\omega_b} \sum_{\mathbf{q}\nu \in b} w_\mathbf{q},
\qquad
\langle J \rangle_b = \frac{\sum_{\mathbf{q}\nu \in b} w_\mathbf{q}\, \|\mathbf{J}_{\mathbf{q}\nu}\|}{\sum_{\mathbf{q}\nu \in b} w_\mathbf{q}},
$$

where $w_\mathbf{q}$ are the q-point weights, normalized to sum to one, so that the bars integrate to the number of modes per cell. Only $\|\mathbf{J}\|$ gives a meaningful DOS colouring: in a non-magnetic crystal $\mathbf{J}(-\mathbf{q}) = -\mathbf{J}(\mathbf{q})$ by time-reversal symmetry, so a signed component averages to zero over the zone, and on a symmetry-reduced mesh depends on which of $\pm\mathbf{q}$ is stored. The script warns when a signed component is combined with a DOS.

## Installation

```bash
git clone https://github.com/<user>/phonopy-chirality.git
cd phonopy-chirality
pip install -r requirements.txt
```

The plotting script needs only NumPy, Matplotlib and h5py (plus PyYAML for `band.yaml` input). Phonopy itself is needed only to generate the input files, and SeeK-path only for the band-path helper below.

## Preparing Phonopy output

Two files are needed, both with eigenvectors: a band structure along a high-symmetry path, and a regular q-mesh for the DOS. The mesh is optional; without it the DOS is built from the path q-points alone, which samples high-symmetry lines rather than the whole Brillouin zone (BZ), and the panel is labelled "Path DOS".

### Command line

With force constants available (for example in `phonopy_params.yaml`):

```
# band.conf
BAND = 0 0 0  1/2 0 0  1/2 1/2 0,  1/4 1/4 1/4  0 0 0
BAND_LABELS = $\Gamma$ X M SIGMA_0 $\Gamma$
BAND_POINTS = 101
BAND_CONNECTION = .TRUE.
EIGENVECTORS = .TRUE.
HDF5 = .TRUE.
```

```
# mesh.conf
MESH = 16 16 16
EIGENVECTORS = .TRUE.
HDF5 = .TRUE.
```

```bash
phonopy-load phonopy_params.yaml --config band.conf    # writes band.hdf5
phonopy-load phonopy_params.yaml --config mesh.conf    # writes mesh.hdf5
```

`BAND_LABELS` must contain exactly one label per breakpoint of the path: one per point, counting both sides of a discontinuity (the comma in `BAND`). If the count is wrong, Phonopy drops the labels without warning; `--labels` in the plotting script supplies them afterwards.

Mesh eigenvectors occupy $N_q \times (3N_\mathrm{atom})^2$ complex numbers over the irreducible q-points; for a 30-atom orthorhombic cell a 30×30×30 mesh needs roughly 450 MB, and a 16×16×16 mesh (about 70 MB) is usually sufficient for the DOS.

### Python API

Scripts that drive Phonopy directly need four things: eigenvectors, the path split into segments with `path_connections`, labels counted per breakpoint, and the HDF5 writers.

```python
phonon.run_band_structure(
    paths,                          # list of (npoints, 3) arrays, one per segment
    with_eigenvectors=True,
    is_band_connection=True,
    path_connections=connections,   # False where the path jumps to a new point
    labels=labels,                  # one per breakpoint, i.e. sum(2 - c for c in connections)
)
phonon.write_hdf5_band_structure(filename="band.hdf5")

phonon.run_mesh([16, 16, 16], with_eigenvectors=True)
phonon.write_hdf5_mesh()            # always writes ./mesh.hdf5; change directory first if needed
```

Passing the whole path as a single segment (`paths = [all_points]`) still produces a valid `band.hdf5`, but Phonopy then has no breakpoints to label.

#### Generating the band path

The function below builds the SeeK-path (HPKOT) path and returns it in the form `run_band_structure` expects, expressed in the basis of Phonopy's primitive cell:

```python
import numpy as np
import seekpath


def seekpath_band_path(primitive, npoints=101):
    """SeeK-path (HPKOT) band path in the basis of Phonopy's primitive cell.

    Returns (paths, path_connections, labels) for Phonopy.run_band_structure.
    """
    A = np.asarray(primitive.cell)
    res = seekpath.get_path((A, primitive.scaled_positions, primitive.numbers),
                            with_time_reversal=True)
    # SeeK-path standardizes, and may rotate, the cell. Undo the rotation and
    # express its reciprocal basis in Phonopy's: q_phonopy = q_seekpath @ inv(M).T
    M = np.array(res["primitive_lattice"]) @ np.array(res["rotation_matrix"]) @ np.linalg.inv(A)
    to_phonopy = np.linalg.inv(M).T
    coords = {k: np.array(v) @ to_phonopy for k, v in res["point_coords"].items()}

    paths, connections, labels = [], [], []
    for i, (start, end) in enumerate(res["path"]):
        if i == 0 or res["path"][i - 1][1] != start:   # a new continuous piece
            if connections:
                connections[-1] = False
            labels.append(start)
        paths.append(np.linspace(coords[start], coords[end], npoints))
        connections.append(True)
        labels.append(end)
    connections[-1] = False
    return paths, connections, labels


paths, connections, labels = seekpath_band_path(phonon.primitive)
```

High-symmetry coordinates from SeeK-path, or from pymatgen's `HighSymmKpath` and `KPathSeek`, are fractional coordinates of *their own* standardized primitive cell, which generally differs from Phonopy's primitive cell in basis and orientation. Passing them to Phonopy unchanged places the path at the wrong points of the BZ, often without any obvious sign in the band structure. Mapping them through the lattice of a *different* standard primitive cell, such as `SpacegroupAnalyzer.get_primitive_standard_structure()`, works only for some lattices: it holds for orthorhombic and tetragonal cells aligned with the Cartesian axes, but fails for rotated, hexagonal, monoclinic and triclinic cells. The function above uses SeeK-path's own cell and rotation, and is correct in each of these cases.

### Scripts generated by uMLIP-Interactive

Phonon scripts generated by [uMLIP-Interactive](https://github.com/bracerino/uMLIP-Interactive) (checked against a script generated in May 2026) build the band path as a single segment, compute the mesh without eigenvectors, and write neither HDF5 file. The changes needed, in the phonon section of the generated script's `main()`, are:

1. **Band path.** Replace the construction of the single concatenated path (`bands = [all_kpts]`) with segment lists, connections and labels from `seekpath_band_path(phonon.primitive, npoints_per_segment)` above. This also corrects the path coordinates, which are otherwise given in pymatgen's standardized basis rather than Phonopy's.
2. **Band structure.** Add `path_connections=connections` and `labels=labels` to the existing `phonon.run_band_structure(...)` call, which already requests eigenvectors, and write the result with `phonon.write_hdf5_band_structure(filename=str(out_folder / "band.hdf5"))`.
3. **Mesh.** Change `phonon.run_mesh(dos_mesh, with_eigenvectors=False)` to `with_eigenvectors=True`, and after `out_folder` is created call `phonon.write_hdf5_mesh()` from inside it (for example, `os.chdir(out_folder)` and back). Reduce `dos_mesh` for large cells, as noted above.
4. **Quick-look plot.** The script's own band plot reads `band_dict["frequencies"][0]`, which with a segmented path holds only the first segment. Concatenate the segments instead: `np.concatenate(band_dict["frequencies"], axis=0)`.

The machine-learning potentials in these scripts run through PyTorch or TensorFlow, which parallelize with threads rather than MPI; on a cluster, run one task per node and set `OMP_NUM_THREADS` and `torch.set_num_threads()` to match the allocated cores.

## Plotting

```bash
# Band structure coloured by |J|
python plot_chiral_bands.py band.hdf5

# With a DOS panel (mesh.hdf5 in the same directory is found automatically)
python plot_chiral_bands.py band.hdf5 --dos

# DOS on its own
python plot_chiral_bands.py mesh.hdf5

# Signed component, in meV, cropped
python plot_chiral_bands.py band.hdf5 --component z --unit meV --ylim 0 50
```

`band.yaml` files are also accepted (run Phonopy with `EIGENVECTORS = .TRUE.`); the parsed arrays are cached to `band.yaml.npz`, so later runs skip the slow YAML parser.

### Colour scale

Phonon angular momentum is usually concentrated near zero with a thin tail of strongly chiral modes, which leaves almost every band in the darkest part of a linear 0–1 colour scale. The script prints the percentiles of the data on every run so that limits can be chosen sensibly:

| Option | Effect |
|---|---|
| `--vmin`, `--vmax` | Fixed colour limits (defaults: 0 and 1 for $\|\mathbf{J}\|$; symmetric for signed components) |
| `--clim-pct P` | Upper limit set to the P-th percentile of the data |
| `--cscale power --cgamma 0.4` | Power-law mapping that expands the low-$J$ end; symmetric about zero for signed components |
| `--cscale log` | Logarithmic (or symmetric-logarithmic) mapping |
| `--cmap` | Any Matplotlib colormap |

Data-driven limits are ignored when the angular momentum is below $10^{-6}\,\hbar$ everywhere, as in centrosymmetric crystals, where every mode is achiral; rescaling would otherwise render rounding noise as chirality.

### All options

| Option | Default | Description |
|---|---|---|
| `--component {total,x,y,z}` | `total` | Quantity used for colouring |
| `--unit {THz,meV,cm-1}` | `THz` | Frequency unit |
| `--ylim YMIN YMAX` | data range | Frequency range, in the chosen unit |
| `--labels "G X M\|S G"` | from file | Override the tick labels: one per tick, i.e. number of segments + 1, with `X\|Y` at a discontinuity |
| `--no-sort-freq` | off | Keep Phonopy's band connection instead of sorting modes by frequency at each q-point |
| `--dos`, `--dos-only` | off | DOS side panel, or a DOS-only figure |
| `--mesh MESH_HDF5` | auto | Mesh file for the DOS |
| `--dos-bins N` | 150 | Number of histogram bins across the frequency range (the `--ylim` window if given) |
| `--dos-binwidth W` | – | Bin width in the chosen frequency unit; overrides `--dos-bins` |
| `--dos-out CSV` | – | Write bin edges, DOS and mean chirality per bin to CSV |
| `--no-degen-zero`, `--degen-tol` | off, 1e-5 | Degeneracy handling |
| `--fontsize`, `--linewidth`, `--dpi` | 16, 1.5, 300 | Figure styling |
| `--no-cache` | off | Do not read or write the `band.yaml.npz` cache |
| `-o FILE` | `band_chirality.png` | Output figure |

Labels are converted to LaTeX (`GAMMA` and `$\Gamma$` → $\Gamma$, `SIGMA_0` → $\Sigma_0$); crowded labels are deliberately left in place for adjustment in post-processing.

## Implementation notes

- Phonopy stores HDF5 eigenvectors as a complex matrix per q-point, with shape `(npath, nqpp, 3N, 3N)` for bands and `(nq, 3N, 3N)` for the mesh, and with **modes as columns**. The YAML layout differs.
- `is_band_connection=True` can mis-assign bands in crowded spectra, producing spurious crossings. The plotting script sorts modes by frequency at each q-point by default, carrying each mode's angular momentum with it.
