# Daily login setup. Loads modules and selects which uv env to use.
#
# Usage:
#   source login_setup.sh eval   # vllm + transformers>=5.5.0 (fast inference)
#   source login_setup.sh lens   # latentlens

ENV="${1:-}"
if [[ "$ENV" != "eval" && "$ENV" != "lens" ]]; then
    echo "Usage: source login_setup.sh {eval|lens}"
    return 1 2>/dev/null || exit 1
fi

module purge
module load anaconda3/2025.6
conda activate uv

export UV_PROJECT_ENVIRONMENT=".venv-$ENV"
# `uv run` would otherwise re-sync the default groups only, which resolve some packages
# (e.g. numpy) differently from the eval/lens groups the environment was built with.
export UV_NO_SYNC=1

# Ensure conda's libstdc++ is found before the system one
# (needed for CXXABI_1.3.15 required by vllm/sqlite3/ICU)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$(dirname "${BASH_SOURCE[0]}")"

nvidia-smi -L

echo ""
echo "Env: $UV_PROJECT_ENVIRONMENT"
echo "Run: uv run python -m experiments.<name> ..."
