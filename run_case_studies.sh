#!/usr/bin/env bash
# run_case_studies.sh — run time_independent and/or time_dependent case_studies.py,
# optionally sweeping over one or more bEPIC parameters.
#
# Activate the right Python environment (bEPIC + priors + benchmark installed)
# before running this script — it just calls `python` on your PATH.
#
# Usage:
#   ./run_case_studies.sh [options]
#
# Options:
#   --which {both,ti,td}        Which workflow(s) to run (default: both)
#   --case-study NAME           CASE_STUDY to use, e.g. Ferndale/Ridgecrest/ElMayor (default: Ferndale)
#   --sigma-s   "v1 v2 ..."     Sweep sigma_s over these values (space-separated)
#   --edt-sigma-s "v1 v2 ..."   Sweep edt_sigma_s over these values
#   --dtt-weight "v1 v2 ..."    Sweep dtt_weight over these values
#   --max-trigs "v1 v2 ..."     Sweep max_trigs over these values
#   --alpha "v1 v2 ..."         Sweep PRIOR_ALPHA over these values (time-dependent only)
#   -h, --help                  Show this help
#
# All sweep flags are optional; each defaults to a single run with the
# script's built-in default for that parameter. Combining several sweep
# flags runs every combination (Cartesian product).
#
# NOTE: output/figure directories are tagged by whichever of sigma_s /
# edt_sigma_s is being varied (sig_<value> or edt_<value>), and by max_trigs
# (max_trigs_<value>) — these sweeps won't overwrite each other. dtt_weight
# has no directory tag today, so sweeping it alone will overwrite results
# between runs.
#
# Examples:
#   ./run_case_studies.sh --sigma-s "1.0 1.5 2.0"
#   ./run_case_studies.sh --which td --case-study Ridgecrest --alpha "0.1 0.5 1.0"
#   ./run_case_studies.sh --sigma-s "1.0 2.0" --max-trigs "10 15"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WHICH="both"
CASE_STUDY="Ferndale"
SIGMA_S_VALUES=("")
EDT_SIGMA_S_VALUES=("")
DTT_WEIGHT_VALUES=("")
MAX_TRIGS_VALUES=("")
ALPHA_VALUES=("")

# Help function - prints top of file
usage() {
    sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --which)        WHICH="$2"; shift 2 ;;
        --case-study)   CASE_STUDY="$2"; shift 2 ;;
        --sigma-s)      read -ra SIGMA_S_VALUES <<< "$2"; shift 2 ;;
        --edt-sigma-s)  read -ra EDT_SIGMA_S_VALUES <<< "$2"; shift 2 ;;
        --dtt-weight)   read -ra DTT_WEIGHT_VALUES <<< "$2"; shift 2 ;;
        --max-trigs)    read -ra MAX_TRIGS_VALUES <<< "$2"; shift 2 ;;
        --alpha)        read -ra ALPHA_VALUES <<< "$2"; shift 2 ;;
        -h|--help)      usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

case "$WHICH" in
    both|ti|td) ;;
    *) echo "--which must be one of: both, ti, td" >&2; exit 1 ;;
esac

# Vary the output directory tag to match whichever of sigma_s / edt_sigma_s
# actually has more than one value in this sweep.
if [[ ${#EDT_SIGMA_S_VALUES[@]} -gt 1 ]]; then
    export VARY_EDT=1
    export VARY_SIG=0
else
    export VARY_EDT=0
    export VARY_SIG=1
fi

export CASE_STUDY

run_one() {
    local sigma="$1" edt="$2" dtt="$3" mt="$4" alpha="$5"

    # one line if statement
    # Syntax: [[ IF ]] && then do this || else do this
    # If string is not empty && Export this variable to os  || else unset the variable to prevent leakage in future runs
    [[ -n "$sigma" ]] && export BENCHMARK_SIGMA_S="$sigma"     || unset -v BENCHMARK_SIGMA_S
    [[ -n "$edt"   ]] && export BENCHMARK_EDT_SIGMA_S="$edt"   || unset -v BENCHMARK_EDT_SIGMA_S
    [[ -n "$dtt"   ]] && export BENCHMARK_DTT_WEIGHT="$dtt"    || unset -v BENCHMARK_DTT_WEIGHT
    [[ -n "$mt"    ]] && export BENCHMARK_MAX_TRIGS="$mt"      || unset -v BENCHMARK_MAX_TRIGS
    [[ -n "$alpha" ]] && export PRIOR_ALPHA="$alpha"           || unset -v PRIOR_ALPHA

    echo "=== case_study=$CASE_STUDY sigma_s=${sigma:-default} edt_sigma_s=${edt:-default} dtt_weight=${dtt:-default} max_trigs=${mt:-default} alpha=${alpha:-default} ==="

    if [[ "$WHICH" == "both" || "$WHICH" == "ti" ]]; then
        echo "--- time_independent_scripts/case_studies.py ---"
        python "$SCRIPT_DIR/time_independent_scripts/case_studies.py"
    fi
    if [[ "$WHICH" == "both" || "$WHICH" == "td" ]]; then
        echo "--- time_dependent_scripts/case_studies.py ---"
        python "$SCRIPT_DIR/time_dependent_scripts/case_studies.py"
    fi
}

for sigma in "${SIGMA_S_VALUES[@]}"; do
for edt in "${EDT_SIGMA_S_VALUES[@]}"; do
for dtt in "${DTT_WEIGHT_VALUES[@]}"; do
for mt in "${MAX_TRIGS_VALUES[@]}"; do
for alpha in "${ALPHA_VALUES[@]}"; do
    run_one "$sigma" "$edt" "$dtt" "$mt" "$alpha"
done
done
done
done
done
