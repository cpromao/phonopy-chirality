#!/usr/bin/env bash
# Run plot_chiral_bands.py in every directory below ROOT that holds a band file.
#
# Usage:
#   plot_all.sh [ROOT] [plot options]      # default ROOT: current directory
#
# Plot options are added after the defaults below; where an option is given
# twice the later one wins, so e.g. `--ylim 0 30` overrides the default range.
#   plot_all.sh results --dpi 600 --dos-scale separate
#
# In each directory with band.hdf5 or band.yaml (band.hdf5 preferred when both
# exist), runs
#   python plot_chiral_bands.py <band file> --dos --ylim -10 50
# from inside that directory, so band_chirality.png is written next to the
# input and mesh.hdf5 there is picked up for the DOS. Directories are processed
# one at a time, and a failure in one does not stop the rest.
#
# Environment overrides:
#   PLOT_SCRIPT  path to plot_chiral_bands.py
#                (default: ~/python/phonopy-chirality/plot_chiral_bands.py)
#   PYTHON       Python interpreter (default: python)

set -u

PLOT_SCRIPT="${PLOT_SCRIPT:-$HOME/python/phonopy-chirality/plot_chiral_bands.py}"
PYTHON="${PYTHON:-python}"
PLOT_ARGS=(--dos --ylim -10 50)

ROOT="."
if [[ $# -gt 0 && "$1" != -* ]]; then
    ROOT="$1"
    shift
fi
[[ $# -gt 0 && "$1" == "--" ]] && shift     # optional separator
PLOT_ARGS+=("$@")

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "Error: plot script not found: $PLOT_SCRIPT (set PLOT_SCRIPT)" >&2
    exit 1
fi
if [[ ! -d "$ROOT" ]]; then
    echo "Error: not a directory: $ROOT" >&2
    exit 1
fi
PLOT_SCRIPT="$(cd "$(dirname "$PLOT_SCRIPT")" && pwd)/$(basename "$PLOT_SCRIPT")"

# Directories containing a band file, skipping .git folders; NUL-separated so
# paths with spaces are handled.
mapfile -d '' DIRS < <(
    find "$ROOT" -name .git -prune -o -type f \( -name band.hdf5 -o -name band.yaml \) \
         -printf '%h\0' | sort -zu
)

if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "No band.hdf5 or band.yaml found under $ROOT"
    exit 0
fi

echo "Found ${#DIRS[@]} director$([[ ${#DIRS[@]} -eq 1 ]] && echo y || echo ies) under $ROOT"
echo "Plot options: ${PLOT_ARGS[*]}"

ok=()
failed=()
for dir in "${DIRS[@]}"; do
    if [[ -f "$dir/band.hdf5" ]]; then
        band=band.hdf5
    else
        band=band.yaml
    fi
    echo
    echo "=== $dir ($band)"
    if (cd "$dir" && "$PYTHON" "$PLOT_SCRIPT" "$band" "${PLOT_ARGS[@]}"); then
        ok+=("$dir")
    else
        failed+=("$dir")
    fi
done

echo
echo "Done: ${#ok[@]} succeeded, ${#failed[@]} failed."
for dir in "${failed[@]}"; do
    echo "  FAILED: $dir"
done
[[ ${#failed[@]} -eq 0 ]]
