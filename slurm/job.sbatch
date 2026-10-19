#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Generic worker: runs one experiment module inside the selected uv environment.
#
# Usage:
#   sbatch --job-name=... --gres=gpu:1 --time=... slurm/job.sbatch <eval|lens> <module> [args...]
# Example:
#   sbatch --gres=gpu:1 --time=2:00:00 slurm/job.sbatch eval experiments.icl_scale \
#       --model_path "$VLM_ICL_MODELS/Qwen3-VL-4B-Instruct" --tasks counting
#
# Resources (job name, gres, time, mem, partition, ...) are passed on the sbatch
# command line; slurm/launch.sh builds these calls for the paper experiments.
# Repository root: $VLM_ICL_ROOT if set, otherwise the directory sbatch was run from.

ENV_NAME="${1:?usage: job.sbatch <eval|lens> <module> [args...]}"
MODULE="${2:?usage: job.sbatch <eval|lens> <module> [args...]}"
shift 2

ROOT="${VLM_ICL_ROOT:-$SLURM_SUBMIT_DIR}"

source "$ROOT/login_setup.sh" "$ENV_NAME"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=ERROR

echo "========================================================================"
echo "Module: ${MODULE}"
echo "Args:   $*"
echo "Started at: $(date)"
echo "========================================================================"

uv run python -m "$MODULE" "$@"
STATUS=$?

echo "Completed at: $(date) (exit ${STATUS})"
exit "$STATUS"
