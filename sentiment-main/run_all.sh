#!/bin/bash
# ============================================================
# run_all.sh — Master orchestrator for vast.ai pipeline
# ============================================================
# Runs setup → frozen cache → training in sequence.
# Saves timestamped logs for every stage. Stops on failure.
#
# Usage:
#   chmod +x run_all.sh && ./run_all.sh
#
# Or skip setup if already done:
#   ./run_all.sh --skip-setup
#
# Or resume from training only (cache already built):
#   ./run_all.sh --skip-setup --skip-cache
# ============================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
WORK_DIR="/workspace"
CODE_DIR="${WORK_DIR}/cached_pipeline"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_DIR="${WORK_DIR}/logs"
IEMOCAP_PATH="${DATA_DIR}/IEMOCAP_full_release"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────────────────────
# PARSE FLAGS
# ─────────────────────────────────────────────────────────────
SKIP_SETUP=false
SKIP_CACHE=false

for arg in "$@"; do
    case $arg in
        --skip-setup)  SKIP_SETUP=true ;;
        --skip-cache)  SKIP_CACHE=true ;;
        --help|-h)
            echo "Usage: ./run_all.sh [--skip-setup] [--skip-cache]"
            echo "  --skip-setup  Skip environment setup (deps already installed)"
            echo "  --skip-cache  Skip frozen cache build (cache already exists)"
            exit 0
            ;;
        *)  echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────
# SETUP LOGGING
# ─────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

SETUP_LOG="${LOG_DIR}/01_setup_${TIMESTAMP}.log"
CACHE_LOG="${LOG_DIR}/02_cache_${TIMESTAMP}.log"
TRAIN_LOG="${LOG_DIR}/03_train_${TIMESTAMP}.log"
MASTER_LOG="${LOG_DIR}/00_master_${TIMESTAMP}.log"

# Log to both file and stdout
log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "${MASTER_LOG}"
}

# Run a command, stream to stdout AND log file, abort on failure
run_stage() {
    local stage_name="$1"
    local log_file="$2"
    shift 2

    log "━━━ STAGE: ${stage_name} ━━━"
    log "  Command: $*"
    log "  Log:     ${log_file}"

    local start_time=$(date +%s)

    # Run with tee so output goes to both terminal and log
    if "$@" 2>&1 | tee -a "${log_file}"; then
        local end_time=$(date +%s)
        local elapsed=$(( end_time - start_time ))
        local mins=$(( elapsed / 60 ))
        local secs=$(( elapsed % 60 ))
        log "✅ ${stage_name} completed in ${mins}m ${secs}s"
        return 0
    else
        local exit_code=$?
        log "❌ ${stage_name} FAILED (exit code ${exit_code})"
        log "   Check log: ${log_file}"
        log "   Last 20 lines:"
        tail -20 "${log_file}" | tee -a "${MASTER_LOG}"
        exit ${exit_code}
    fi
}

# ─────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────
echo "" | tee -a "${MASTER_LOG}"
echo "╔══════════════════════════════════════════════════╗" | tee -a "${MASTER_LOG}"
echo "║  Quantum Enhanced Fusion — Full Pipeline        ║" | tee -a "${MASTER_LOG}"
echo "║  $(date '+%Y-%m-%d %H:%M:%S')                          ║" | tee -a "${MASTER_LOG}"
echo "╠══════════════════════════════════════════════════╣" | tee -a "${MASTER_LOG}"
echo "║  Skip setup: ${SKIP_SETUP}                            ║" | tee -a "${MASTER_LOG}"
echo "║  Skip cache: ${SKIP_CACHE}                            ║" | tee -a "${MASTER_LOG}"
echo "║  Logs:       ${LOG_DIR}/            ║" | tee -a "${MASTER_LOG}"
echo "╚══════════════════════════════════════════════════╝" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

PIPELINE_START=$(date +%s)

# ─────────────────────────────────────────────────────────────
# STAGE 1: ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────
if [ "${SKIP_SETUP}" = true ]; then
    log "⏭️  Skipping setup (--skip-setup)"
else
    run_stage "Environment Setup" "${SETUP_LOG}" \
        bash "${CODE_DIR}/setup_vastai.sh"
fi

# ─────────────────────────────────────────────────────────────
# STAGE 2: BUILD FROZEN CACHE
# ─────────────────────────────────────────────────────────────
if [ "${SKIP_CACHE}" = true ]; then
    log "⏭️  Skipping cache build (--skip-cache)"
elif [ -f "${OUTPUT_DIR}/frozen_features.pt" ]; then
    log "⏭️  Frozen cache already exists at ${OUTPUT_DIR}/frozen_features.pt"
    log "   Delete it to rebuild, or use --skip-cache"
else
    run_stage "Frozen Cache Build" "${CACHE_LOG}" \
        python "${CODE_DIR}/build_frozen_cache.py" \
            --iemocap_path "${IEMOCAP_PATH}" \
            --save_dir "${OUTPUT_DIR}"
fi

# Verify cache exists before training
if [ ! -f "${OUTPUT_DIR}/frozen_features.pt" ]; then
    log "❌ FATAL: frozen_features.pt not found in ${OUTPUT_DIR}"
    log "   Run without --skip-cache, or check cache build logs."
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# STAGE 3: TRAINING (5-fold CV)
# ─────────────────────────────────────────────────────────────
run_stage "Training (5-fold CV)" "${TRAIN_LOG}" \
    python "${CODE_DIR}/train_single_gpu.py" \
        --cache_path "${OUTPUT_DIR}/frozen_features.pt" \
        --iemocap_path "${IEMOCAP_PATH}" \
        --save_dir "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo "" | tee -a "${MASTER_LOG}"
echo "╔══════════════════════════════════════════════════╗" | tee -a "${MASTER_LOG}"
echo "║  ✅ PIPELINE COMPLETE                           ║" | tee -a "${MASTER_LOG}"
echo "║  Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s                          ║" | tee -a "${MASTER_LOG}"
echo "╠══════════════════════════════════════════════════╣" | tee -a "${MASTER_LOG}"
echo "║  Outputs:                                       ║" | tee -a "${MASTER_LOG}"
echo "║    ${OUTPUT_DIR}/frozen_features.pt   ║" | tee -a "${MASTER_LOG}"
echo "║    ${OUTPUT_DIR}/cached_qfl_fold*.pth ║" | tee -a "${MASTER_LOG}"
echo "║  Logs:                                          ║" | tee -a "${MASTER_LOG}"
echo "║    ${MASTER_LOG}  ║" | tee -a "${MASTER_LOG}"
echo "╚══════════════════════════════════════════════════╝" | tee -a "${MASTER_LOG}"

# List all output files
echo "" | tee -a "${MASTER_LOG}"
log "Output files:"
ls -lh "${OUTPUT_DIR}"/ 2>/dev/null | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"
log "Log files:"
ls -lh "${LOG_DIR}"/ 2>/dev/null | tee -a "${MASTER_LOG}"
