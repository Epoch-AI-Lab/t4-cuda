#!/usr/bin/env bash
# scripts/run_kaggle_chalk7b.sh
#
# ONE-SHOT Big-Chalk runner for Kaggle (GPU T4 x2). End-to-end:
#   deps -> repo -> dataset repair -> SFT -> eval -> package
#
# Designed to be RERUNNABLE and state-independent:
#   * every path is absolute (Kaggle resets CWD between cells)
#   * a wiped /kaggle/working is rebuilt automatically
#   * a partial clone or missing dataset is repaired, not fatal
#   * falls back to 1 GPU + QLoRA if only one T4 is attached
#   * full log streamed to /kaggle/working/run_chalk7b.log
#
# Usage:  bash scripts/run_kaggle_chalk7b.sh
set -euo pipefail

REPO_URL="https://github.com/Epoch-AI-Lab/t4-cuda.git"
BRANCH="feat-big-chalk-math-7b"
COMMIT="47911c8f25d0e37bc9840c36c5573a887939c08f"

WORK="${WORK:-/kaggle/working}"
REPO="$WORK/t4-cuda"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B}"
SEED_REL="data/chalk_seeds_500.jsonl"
SEED="$REPO/$SEED_REL"
SEED_LINES="${SEED_LINES:-605}"
SEED_MD5="${SEED_MD5:-3fb09048b62154525c23f74297a480e3}"

OUT="${OUT:-$WORK/big_chalk_7b_sft}"
EVAL_OUT="${EVAL_OUT:-$WORK/big_chalk_7b_eval_results.json}"
TARBALL="${TARBALL:-$WORK/big_chalk_7b_adapter.tar.gz}"
LOG="${LOG:-$WORK/run_chalk7b.log}"
DRY_RUN="${DRY_RUN:-0}"
# TEST_MODE=1 is for local CI only: skips GPU + dependency probes that cannot
# pass on a non-Kaggle box. Never set it in the real run.
TEST_MODE="${TEST_MODE:-0}"
if [ "$TEST_MODE" = "1" ]; then
    SKIP_GPU_CHECK=1
    SKIP_DEPS_CHECK=1
fi

LR="${LR:-1.5e-4}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
# Progress prints come from the trainer. Unbuffered so they stream through tee,
# which is what keeps the Kaggle cell from looking frozen.
export PYTHONUNBUFFERED="1"

ts() { date -u +%H:%M:%S; }
log() { echo "[$(ts)] $*"; }
die() {
    log "FATAL: $*"
    log "----- last 40 log lines -----"
    tail -40 "$LOG" 2>/dev/null || true
    exit 1
}
on_err() {
    log "ABORTED at line $1 (rc=$2)"
    log "----- last 40 log lines -----"
    tail -40 "$LOG" 2>/dev/null || true
    exit "$2"
}
trap 'on_err "$LINENO" "$?"' ERR

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

seed_ok() {
    [ -s "$SEED" ] || return 1
    [ "$(wc -l < "$SEED" 2>/dev/null | tr -d ' ')" = "$SEED_LINES" ] || return 1
    [ "$(md5sum "$SEED" | cut -d' ' -f1)" = "$SEED_MD5" ]
}

log "=============== BIG CHALK 7B :: START ==============="

# ---------- 1. hardware ----------
log "[1/6] hardware probe"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || log "nvidia-smi unavailable"
NGPU="$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
log "visible CUDA devices: $NGPU"
if [ "$NGPU" -lt 1 ] && [ "$SKIP_GPU_CHECK" != "1" ]; then
    die "No CUDA GPU visible. In Kaggle set Accelerator = 'GPU T4 x2' and re-run this cell."
fi
QLORA=""
if [ "$NGPU" -lt 2 ] && [ "$SKIP_GPU_CHECK" != "1" ]; then
    log "WARNING: only $NGPU GPU. 7B fp16 needs ~16GB -> enabling QLoRA (--load_in_4bit)."
    QLORA="--load_in_4bit"
elif [ "$NGPU" -ge 2 ]; then
    log "Dual T4 detected -> fp16 + device_map=auto (14 layers/GPU)."
fi

# ---------- 2. deps ----------
log "[2/6] dependencies"
pip install -q --upgrade transformers peft accelerate bitsandbytes datasets 2>&1 | tail -5 || true
if [ "${SKIP_DEPS_CHECK:-0}" != "1" ]; then
    python3 -c 'import torch,transformers,peft,accelerate;print("  torch",torch.__version__,"| transformers",transformers.__version__,"| peft",peft.__version__)' \
        || die "dependency import failed after pip install"
fi

# ---------- 3. repo ----------
log "[3/6] repository"
if [ -f "$REPO/benchmarks/train_math_sft.py" ]; then
    log "  existing tree found, syncing to $BRANCH"
    git -C "$REPO" fetch --depth 1 origin "$BRANCH" >/dev/null 2>&1 || true
    git -C "$REPO" checkout -q "$BRANCH" 2>/dev/null || true
else
    log "  cloning $BRANCH"
    rm -rf "$REPO"
    git clone -q --depth 1 -b "$BRANCH" "$REPO_URL" "$REPO" \
        || { log "  shallow clone failed, retrying full"; rm -rf "$REPO"; git clone -q -b "$BRANCH" "$REPO_URL" "$REPO"; }
fi
[ -f "$REPO/benchmarks/train_math_sft.py" ] \
    || die "incomplete clone: $REPO/benchmarks/train_math_sft.py not found (check Internet: On)"
log "  HEAD: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# ---------- 4. dataset (the thing that has been breaking) ----------
log "[4/6] dataset"
if seed_ok; then
    log "  dataset OK ($SEED_LINES records, md5 verified)"
else
    log "  dataset missing or corrupt -> repairing"
    mkdir -p "$REPO/data"
    git -C "$REPO" checkout "$BRANCH" -- "$SEED_REL" >/dev/null 2>&1 || true
    if ! seed_ok; then
        log "  git restore did not satisfy checks -> fetching raw blob from $COMMIT"
        curl -fsSL --retry 3 --retry-delay 2 \
            -o "$SEED" \
            "https://raw.githubusercontent.com/Epoch-AI-Lab/t4-cuda/$COMMIT/$SEED_REL" \
            || die "could not download $SEED_REL"
    fi
    seed_ok || die "dataset still invalid after repair ($(wc -l < "$SEED" 2>/dev/null || echo 0) lines)"
    log "  dataset repaired ($SEED_LINES records, md5 verified)"
fi

# ---------- 5. train ----------
if [ "$DRY_RUN" = "1" ]; then
    log "=============== BIG CHALK 7B :: DRY RUN COMPLETE ==============="
    log "environment verified. repo=$REPO"
    log "dataset=$SEED ($SEED_LINES records, md5 ok)"
    log "re-run with DRY_RUN=0 to train."
    exit 0
fi

log "[5/6] cold-start SFT"
log "  model=$MODEL epochs=$EPOCHS lr=$LR out=$OUT"
log "  (unbuffered stdout: progress appears live; do NOT kill this cell)"
# -u is REQUIRED: stdout is a pipe into tee, so CPython would otherwise
# block-buffer every print and the notebook would look frozen for hours.
python3 -u "$REPO/benchmarks/train_math_sft.py" \
    --model_name_or_path "$MODEL" \
    --data_path "$SEED" \
    --output_dir "$OUT" \
    --epochs "$EPOCHS" \
    --batch_size 1 \
    --grad_accum 16 \
    --lr "$LR" \
    --max_length 2048 \
    --lora_r 32 \
    --lora_alpha 64 \
    --logging_steps 5 \
    --max_steps "$MAX_STEPS" \
    $QLORA

[ -f "$OUT/lora_adapter/adapter_model.safetensors" ] || [ -d "$OUT/lora_adapter" ] \
    || die "training finished but no LoRA adapter in $OUT/lora_adapter"
log "  adapter saved"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

# ---------- 6. eval + package ----------
log "[6/6] contest evaluation"
python3 -u "$REPO/benchmarks/eval_math_benchmark.py" \
    --model_name_or_path "$MODEL" \
    --adapter_path "$OUT/lora_adapter" \
    --benchmark_path "$REPO/data/external_math_eval.json" \
    --output_path "$EVAL_OUT" \
    --max_new_tokens 2048 \
    $QLORA

tar -czf "$TARBALL" -C "$OUT" lora_adapter

log "=============== BIG CHALK 7B :: DONE ==============="
log "adapter   : $OUT/lora_adapter"
log "tarball   : $TARBALL"
log "eval json : $EVAL_OUT"
if [ -f "$OUT/training_summary.json" ]; then
    log "----- training_summary.json -----"
    cat "$OUT/training_summary.json"
fi
if [ -f "$EVAL_OUT" ]; then
    log "----- eval results -----"
    cat "$EVAL_OUT"
fi
log "full log  : $LOG"
