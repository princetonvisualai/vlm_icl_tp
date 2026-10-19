#!/usr/bin/env bash
# One-time environment setup. Creates .venv-eval/ or .venv-lens/.
#
# Usage:
#   bash env_setup.sh eval   # vllm + transformers>=5.5.0 (fast inference)
#   bash env_setup.sh lens   # latentlens + transformers

set -eo pipefail

ENV="${1:-}"
if [[ "$ENV" != "eval" && "$ENV" != "lens" ]]; then
    echo "Usage: bash env_setup.sh {eval|lens}"
    exit 1
fi

module purge
module load anaconda3/2025.6
conda activate uv

export UV_PROJECT_ENVIRONMENT=".venv-$ENV"
uv sync --no-default-groups --group "$ENV" --group dev

echo ""
echo "=== Done! ==="
echo "Activate with: source login_setup.sh $ENV"
