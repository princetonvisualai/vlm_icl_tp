#!/bin/bash
# Submit paper experiments to SLURM.
#
# Usage:
#   VLM_ICL_MODELS=/path/to/models bash slurm/launch.sh [options] <experiment> <model> [tasks...] [-- extra args]
#
# Environment:
#   VLM_ICL_MODELS     directory holding the model checkpoints and LatentLens indices (required)
#   VLM_ICL_PARTITION  value for --partition (optional)
#   VLM_ICL_MAIL       value for --mail-user, with --mail-type=END,FAIL (optional)
#   VLM_ICL_ROOT       repository root used by the worker (default: parent of this directory)
#
# Options:
#   --dry-run             print the sbatch commands instead of submitting them
#   --short               1-hour jobs: one job per split for the large models (icl_* experiments)
#   --splits "a b"        run one job per listed split (icl_* experiments)
#   --time T --mem M --constraint C   override the resources of every job
#   --lens-type T         latentlens or logitlens (lens_visualize; default: both)
#   --distract-cue X --expected-bias Y --prior-cue Z   lens experiment cues
#
# Experiments: see the table below; models: see slurm/models.tsv.
# Arguments after "--" are appended to the module's command line.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_TSV="$HERE/models.tsv"
WORKER="$HERE/job.sbatch"

# name|job prefix|env|module (+ fixed leading args)|kind|backend|default tasks|sweep|pre-args|fixed args|time small|time large|mem small|mem large
#   kind   icl   shot counts and resources from models.tsv; args: ... --tasks T [--splits S] <pre> --shot_counts <group> <fixed>
#          extra one job per task (and per shot count for sweeps); resources from the columns below
#          lens  LatentLens experiments (HF backend, resources from the columns below)
#   backend vllm adds "--backend vllm --tensor_parallel_size <tp>"; hf adds nothing (single GPU)
#   sweep  none | k_icl (k>0: --icl --num_icl_examples k) | k_icl_sets2 / k_icl_sets4 (adds --num_icl_sample_sets) | k_shift (k>0: --num_icl_examples k)
#   "-" means unset. Lens experiments always request the constraint in LENS_CONSTRAINT.
LENS_CONSTRAINT="gpu80"
ALL_TASKS="counting shape_identification color_identification direction relation"
read -r -d '' EXPERIMENTS <<TABLE || true
icl_scale|fis|eval|experiments.icl_scale|icl|vllm|$ALL_TASKS|none|-|--n_per_label 20 --seed 42|-|-|-|-
icl_stresstest|fist|eval|experiments.icl_scale --preset stresstest|icl|vllm|$ALL_TASKS|none|-|--n_per_label 20 --seed 42|-|-|-|-
icl_unlearn|fiunl|eval|experiments.icl_unlearn|icl|vllm|$ALL_TASKS|none|-|--n_per_label 20 --seed 42|-|-|-|-
icl_reasoning|fir|eval|experiments.icl_reasoning|icl|vllm|$ALL_TASKS|none|--enable_thinking|--n_per_label 20 --seed 42|-|-|-|-
lens_visualize|llens|lens|experiments.lens_visualize|lens|hf|color_identification shape_identification counting|none|-|--dataset visual_reasoning_lens --icl_counts 4 16 --n_per_label 10 --seed 42|1:00:00|1:00:00|128G|256G
lens_ablate_tokens|fabat|lens|experiments.lens_ablate_tokens|lens|hf|shape_identification color_identification counting|none|-|--output_dir output --n_per_label 20 --num_icl_examples 16 --seed 42|1:00:00|1:00:00|32G|64G
lens_divert|fid|lens|experiments.lens_divert|lens|hf|shape_identification color_identification counting|none|-|--output_dir output --n_per_label 10 --num_icl_examples 8 --seed 42|1:00:00|1:00:00|64G|128G
lens_rewrite|ficr|lens|experiments.lens_rewrite|lens|hf|shape_identification color_identification|none|-|--output_dir output --layers 1 --n_per_label 10 --num_icl_examples 8 --seed 42|1:00:00|1:00:00|64G|256G
answer_distribution|ansdist|eval|experiments.extra.answer_distribution|extra|vllm|counting direction relation|k_icl|-|--output_dir output|4:00:00|8:00:00|32G|64G
answer_distribution_shift|ansshift|eval|experiments.extra.answer_distribution_shift|extra|vllm|counting direction relation|k_shift|-|--output_dir output|4:00:00|8:00:00|32G|64G
control_accuracy|ctrlacc|eval|experiments.extra.control_accuracy|extra|vllm|counting direction relation|none|-|--output_dir output --sample_per_label 50 --num_runs 10 --k_values 1 2 4 8 16 32|24:00:00|24:00:00|16G|16G
control_output_dist|ctrlod|eval|experiments.extra.control_output_dist|extra|vllm|counting direction relation|none|-|--output_dir output --k_values 1 2 4 8 16 32|6:00:00|12:00:00|16G|16G
control_attention_distribution|ctrlattn|eval|experiments.extra.control_attention_distribution|extra|hf|counting direction relation|none|-|--k_values 1 2 4 8 16 --output_dir ./output_vis|1:00:00|1:00:00|-|-
control_hidden_emb|ctrlhemb|eval|experiments.extra.control_hidden_emb|extra|hf|counting direction relation|none|-|--sample_per_label 10 --k_values 1 2 4 8 16 32|12:00:00|12:00:00|-|-
hidden_embedding|hemb|eval|experiments.extra.hidden_embedding|extra|hf|counting direction relation object_reid|none|-|--sample_per_split 50 --num_icl_sample_sets 1|24:00:00|48:00:00|-|-
linear_probe|lprobe|eval|experiments.extra.linear_probe|extra|hf|counting direction relation|none|-|--dataset visual_reasoning_many --sample_per_split 200 --num_icl_sample_sets 20 --icl_counts 0 2 4 8 16 32|24:00:00|24:00:00|-|-
variance_ablation|var|eval|experiments.extra.variance_ablation|extra|vllm|counting direction relation|k_icl_sets4|-|--output_dir output --num_runs 10 --sample_per_split 50 --k 1|2:00:00|6:00:00|16G|64G
evaluate|eval|eval|experiments.extra.evaluate|extra|vllm|counting direction relation|k_icl_sets2|-|--output_dir output --sample_per_split 50 --num_eval_runs --k 1|4:00:00|8:00:00|64G|128G
attention_map|attn|eval|experiments.extra.inference|extra|hf|counting direction relation|k_icl|-|--backend default --sample_per_split 10 --image_attentionmap|6:00:00|12:00:00|-|-
attention_distribution|attndist|eval|experiments.extra.visualize_attention_distribution|extra|hf|direction|k_icl|-|--sample_per_split 10|2:00:00|12:00:00|-|-
TABLE

die() { echo "launch.sh: $*" >&2; exit 1; }

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

DRY=0 SHORT=0 SPLITS_OPT="" TIME_OPT="" MEM_OPT="" CONSTRAINT_OPT=""
LENS_TYPE_OPT="" DISTRACT_OPT="" BIAS_OPT="" PRIOR_OPT=""
POS=() EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --short) SHORT=1 ;;
        --splits) SPLITS_OPT="$2"; shift ;;
        --time) TIME_OPT="$2"; shift ;;
        --mem) MEM_OPT="$2"; shift ;;
        --constraint) CONSTRAINT_OPT="$2"; shift ;;
        --lens-type) LENS_TYPE_OPT="$2"; shift ;;
        --distract-cue) DISTRACT_OPT="$2"; shift ;;
        --expected-bias) BIAS_OPT="$2"; shift ;;
        --prior-cue) PRIOR_OPT="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) POS+=("$1") ;;
    esac
    shift
done
[[ ${#POS[@]} -ge 2 ]] || { usage >&2; exit 1; }
EXP="${POS[0]}"; MODEL="${POS[1]}"; TASKS_ARG=("${POS[@]:2}")

[[ -n "${VLM_ICL_MODELS:-}" ]] || die "VLM_ICL_MODELS is not set"
export VLM_ICL_ROOT="${VLM_ICL_ROOT:-$(cd "$HERE/.." && pwd)}"

# --- model row -------------------------------------------------------------
MROW="$(awk -F'\t' -v k="$MODEL" 'NR>1 && $1==k' "$MODELS_TSV")"
[[ -n "$MROW" ]] || die "unknown model '$MODEL' (see $MODELS_TSV)"
IFS=$'\t' read -r M_KEY M_PATH M_CLASS M_GRES M_TP M_MEM M_TIME M_TIME_R M_SHOTS M_SHOTS_SHORT \
    M_SPLITJOBS M_SHORTC M_LENS <<<"$MROW"

# --- experiment row --------------------------------------------------------
EROW="$(awk -F'|' -v k="$EXP" '$1==k' <<<"$EXPERIMENTS")"
[[ -n "$EROW" ]] || die "unknown experiment '$EXP' (known: $(cut -d'|' -f1 <<<"$EXPERIMENTS" | tr '\n' ' '))"
IFS='|' read -r E_NAME E_PREFIX E_ENV E_MODULE E_KIND E_BACKEND E_TASKS E_SWEEP E_PRE E_FIXED \
    E_TIME_S E_TIME_L E_MEM_S E_MEM_L <<<"$EROW"
read -ra MODWORDS <<<"$E_MODULE"
read -ra FIXED <<<"$E_FIXED"
PRE=(); [[ "$E_PRE" == "-" ]] || read -ra PRE <<<"$E_PRE"
if [[ ${#TASKS_ARG[@]} -gt 0 ]]; then TASKS=("${TASKS_ARG[@]}"); else read -ra TASKS <<<"$E_TASKS"; fi

# --- helpers -----------------------------------------------------------------
submit() {  # submit <job-name> <time> <mem> <gres> <constraint> <partition-ok> -- <worker args...>
    local name="$1" time="$2" mem="$3" gres="$4" constraint="$5" use_part="$6"
    shift 7
    local cmd=(sbatch --job-name="$name" --gres="$gres" --time="${TIME_OPT:-$time}")
    local m="${MEM_OPT:-$mem}" c="${CONSTRAINT_OPT:-$constraint}"
    [[ "$m" == "-" ]] || cmd+=(--mem="$m")
    [[ "$c" == "-" ]] || cmd+=(--constraint="$c")
    [[ -z "${VLM_ICL_PARTITION:-}" || "$use_part" != 1 ]] || cmd+=(--partition="$VLM_ICL_PARTITION")
    [[ -z "${VLM_ICL_MAIL:-}" ]] || cmd+=(--mail-user="$VLM_ICL_MAIL" --mail-type=END,FAIL)
    cmd+=(--export=ALL "$WORKER" "$E_ENV" "${MODWORDS[0]}" "${MODWORDS[@]:1}" "$@" "${EXTRA[@]}")
    if [[ $DRY == 1 ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

default_splits() {  # default_splits <task>
    if [[ "$EXP" == icl_stresstest ]]; then
        [[ "$1" == counting ]] && echo "simple noprior complex" || echo "complex"
    elif [[ "$EXP" == icl_unlearn && "$MODEL" == gemma4-31b ]]; then
        echo "simple complex"
    else
        echo "simple noprior complex"
    fi
}

MODEL_DIR="$VLM_ICL_MODELS/$M_PATH"

# --- icl experiments (paper scale / stresstest / unlearn / reasoning) -------------
if [[ "$E_KIND" == icl ]]; then
    if [[ $SHORT == 1 ]]; then
        TIME="1:00:00"; SHOTS="$M_SHOTS_SHORT"; TAG="${E_PREFIX}_short"
    else
        TIME="$M_TIME"; SHOTS="$M_SHOTS"; TAG="$E_PREFIX"
        if [[ "$EXP" == icl_reasoning && "$M_TIME_R" != "-" ]]; then TIME="$M_TIME_R"; fi
    fi
    CONSTRAINT="-"; USE_PART=1
    if [[ $SHORT == 1 && "$M_SHORTC" != "-" ]]; then CONSTRAINT="$M_SHORTC"; USE_PART=0; fi
    IFS=';' read -ra GROUPS_ <<<"$SHOTS"
    for task in "${TASKS[@]}"; do
        SPLIT_LIST=("")
        if [[ -n "$SPLITS_OPT" ]]; then read -ra SPLIT_LIST <<<"$SPLITS_OPT"
        elif [[ $SHORT == 1 && "$M_SPLITJOBS" == 1 ]]; then read -ra SPLIT_LIST <<<"$(default_splits "$task")"; fi
        for split in "${SPLIT_LIST[@]}"; do
            for grp in "${GROUPS_[@]}"; do
                read -ra SH <<<"$grp"
                name="${TAG}_${MODEL}_${task}"
                SPLIT_ARGS=()
                [[ -z "$split" ]] || { name+="_${split}"; SPLIT_ARGS=(--splits "$split"); }
                [[ ${#GROUPS_[@]} -eq 1 ]] || name+="_s${SH[-1]}"
                submit "$name" "$TIME" "$M_MEM" "$M_GRES" "$CONSTRAINT" "$USE_PART" -- \
                    --model_path "$MODEL_DIR" --backend vllm --output_dir output \
                    --tensor_parallel_size "$M_TP" --tasks "$task" "${SPLIT_ARGS[@]}" \
                    "${PRE[@]}" --shot_counts "${SH[@]}" "${FIXED[@]}"
            done
        done
    done
    exit 0
fi

# --- resources for extras and lens -------------------------------------------------
if [[ "$M_CLASS" == s ]]; then TIME="$E_TIME_S"; MEM="$E_MEM_S"; else TIME="$E_TIME_L"; MEM="$E_MEM_L"; fi
GRES="$M_GRES"
[[ "$E_KIND" == lens || "$E_BACKEND" == vllm ]] || GRES="gpu:1"
BACKEND_ARGS=()
[[ "$E_BACKEND" != vllm ]] || BACKEND_ARGS=(--backend vllm --tensor_parallel_size "$M_TP")

# --- extras ----------------------------------------------------------------
if [[ "$E_KIND" == extra ]]; then
    read -ra KS <<<"$(tr ';' ' ' <<<"$M_SHOTS")"
    for task in "${TASKS[@]}"; do
        base=(--model_path "$MODEL_DIR" "${BACKEND_ARGS[@]}" --tasks "$task" "${FIXED[@]}")
        if [[ "$E_SWEEP" == none ]]; then
            submit "${E_PREFIX}_${MODEL}_${task}" "$TIME" "$MEM" "$GRES" "-" 1 -- "${base[@]}"
            continue
        fi
        for k in "${KS[@]}"; do
            sweep=()
            case "$E_SWEEP" in
                k_icl)       [[ $k == 0 ]] || sweep=(--icl --num_icl_examples "$k") ;;
                k_icl_sets2) [[ $k == 0 ]] || sweep=(--icl --num_icl_examples "$k" --num_icl_sample_sets 2) ;;
                k_icl_sets4) [[ $k == 0 ]] || sweep=(--icl --num_icl_examples "$k" --num_icl_sample_sets 4) ;;
                k_shift)     [[ $k != 0 ]] || continue; sweep=(--num_icl_examples "$k") ;;
            esac
            submit "${E_PREFIX}_${MODEL}_${task}_k${k}" "$TIME" "$MEM" "$GRES" "-" 1 -- "${base[@]}" "${sweep[@]}"
        done
    done
    exit 0
fi

# --- lens experiments --------------------------------------------------------------
[[ "$M_LENS" != "-" ]] || die "model '$MODEL' has no LatentLens index in models.tsv"
INDEX="$VLM_ICL_MODELS/$M_LENS"
CONSTRAINT="$LENS_CONSTRAINT"
cue_pair() {  # cue_pair <task> -> "distractor bias"
    case "$1" in
        shape_identification) echo "stop octagon" ;;
        color_identification) echo "tiger orange" ;;
        *) echo "" ;;
    esac
}
for task in "${TASKS[@]}"; do
    base=(--model_path "$MODEL_DIR" --latentlens_index "$INDEX" --tasks "$task")
    name="${E_PREFIX}_${MODEL}_${task}"
    fixed=("${FIXED[@]}")
    case "$EXP" in
        lens_visualize)
            for lt in ${LENS_TYPE_OPT:-latentlens logitlens}; do
                extra=()
                [[ "$lt" == latentlens ]] || extra=(--lens_type "$lt" --top_k 50)
                submit "${name}_${lt}" "$TIME" "$MEM" "$GRES" "$CONSTRAINT" 0 -- "${base[@]}" "${extra[@]}" "${fixed[@]}"
            done
            continue ;;
        lens_ablate_tokens)
            cues=()
            [[ -z "$DISTRACT_OPT" ]] || cues+=(--distract_cue "$DISTRACT_OPT")
            [[ -z "$BIAS_OPT" ]] || cues+=(--expected_bias "$BIAS_OPT")
            if [[ "$MODEL" == qwen25vl-3b ]]; then
                read -r d b <<<"$(cue_pair "$task")"
                [[ -n "$d" ]] || die "no default cues for task '$task'; pass --distract-cue/--expected-bias"
                cues=(--distract_cue "${DISTRACT_OPT:-$d}" --expected_bias "${BIAS_OPT:-$b}")
                fixed=(--output_dir output --layers 1 --top_k 5 --n_per_label 10 --num_icl_examples 8 --seed 42)
                TIME="6:00:00"; MEM="32G"; CONSTRAINT="-"
            fi
            submit "$name" "$TIME" "$MEM" "$GRES" "$CONSTRAINT" 0 -- "${base[@]}" "${cues[@]}" "${fixed[@]}" ;;
        lens_divert)
            cues=()
            [[ -z "$PRIOR_OPT" ]] || cues+=(--prior_cue "$PRIOR_OPT")
            submit "$name" "$TIME" "$MEM" "$GRES" "$CONSTRAINT" 0 -- "${base[@]}" "${cues[@]}" "${fixed[@]}" ;;
        lens_rewrite)
            read -r d b <<<"$(cue_pair "$task")"
            d="${DISTRACT_OPT:-$d}"; b="${BIAS_OPT:-$b}"
            [[ -n "$d" && -n "$b" ]] || die "lens_rewrite needs --distract-cue and --expected-bias for task '$task'"
            submit "$name" "$TIME" "$MEM" "$GRES" "$CONSTRAINT" 0 -- "${base[@]}" --distract_cue "$d" --expected_bias "$b" "${fixed[@]}" ;;
    esac
done
