#!/bin/bash
# ============================================================
# run_all.sh — Master orchestrator for vast.ai pipeline
# ============================================================
# Clones repo → setup → full fine-tuning (no caching).
# Saves timestamped logs for every stage. Stops on failure.
#
# Usage (on a fresh vast.ai instance):
#   wget -O run_all.sh https://raw.githubusercontent.com/Rafat-Aman/sentiment/main/run_all.sh
#   chmod +x run_all.sh && ./run_all.sh
#
# Or skip setup if already done:
#   ./run_all.sh --skip-setup
# ============================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
REPO_URL="https://github.com/Rafat-Aman/sentiment.git"
WORK_DIR="/workspace"
REPO_DIR="${WORK_DIR}/sentiment"
CODE_DIR="${REPO_DIR}"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_DIR="${WORK_DIR}/logs"
IEMOCAP_PATH="${DATA_DIR}/IEMOCAP_full_release"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────────────────────
# PARSE FLAGS
# ─────────────────────────────────────────────────────────────
SKIP_SETUP=false

for arg in "$@"; do
    case $arg in
        --skip-setup)  SKIP_SETUP=true ;;
        --help|-h)
            echo "Usage: ./run_all.sh [--skip-setup]"
            echo "  --skip-setup  Skip environment setup (deps already installed)"
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

CLONE_LOG="${LOG_DIR}/00_clone_${TIMESTAMP}.log"
SETUP_LOG="${LOG_DIR}/01_setup_${TIMESTAMP}.log"
TRAIN_LOG="${LOG_DIR}/02_train_${TIMESTAMP}.log"
MASTER_LOG="${LOG_DIR}/00_master_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "${MASTER_LOG}"
}

run_stage() {
    local stage_name="$1"
    local log_file="$2"
    shift 2

    log "━━━ STAGE: ${stage_name} ━━━"
    log "  Command: $*"
    log "  Log:     ${log_file}"

    local start_time=$(date +%s)

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
echo "║  Quantum Enhanced Fusion — Full Fine-Tuning     ║" | tee -a "${MASTER_LOG}"
echo "║  $(date '+%Y-%m-%d %H:%M:%S')                          ║" | tee -a "${MASTER_LOG}"
echo "╠══════════════════════════════════════════════════╣" | tee -a "${MASTER_LOG}"
echo "║  Mode:       Full fine-tuning (no caching)      ║" | tee -a "${MASTER_LOG}"
echo "║  Repo:       ${REPO_URL}  ║" | tee -a "${MASTER_LOG}"
echo "║  Skip setup: ${SKIP_SETUP}                            ║" | tee -a "${MASTER_LOG}"
echo "║  Logs:       ${LOG_DIR}/            ║" | tee -a "${MASTER_LOG}"
echo "╚══════════════════════════════════════════════════╝" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

PIPELINE_START=$(date +%s)

# ─────────────────────────────────────────────────────────────
# STAGE 0: CLONE REPO
# ─────────────────────────────────────────────────────────────
if [ -f "${CODE_DIR}/config.py" ]; then
    log "⏭️  Repo already cloned, pulling latest..."
    git -C "${REPO_DIR}" pull 2>&1 | tee -a "${CLONE_LOG}" || true
else
    log "━━━ STAGE: Git Clone ━━━"
    log "  Cloning ${REPO_URL} → ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}" 2>&1 | tee -a "${CLONE_LOG}"
    log "✅ Repo cloned"
fi

if [ ! -f "${CODE_DIR}/config.py" ]; then
    log "❌ FATAL: config.py not found in ${CODE_DIR}"
    exit 1
fi
log "  Code dir: ${CODE_DIR}"

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
# STAGE 2: FULL FINE-TUNING (5-fold CV)
# ─────────────────────────────────────────────────────────────
run_stage "Training — Full Fine-Tuning (5-fold CV)" "${TRAIN_LOG}" \
    python "${CODE_DIR}/train_full.py" \
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
echo "║    ${OUTPUT_DIR}/full_qfl_fold*.pth  ║" | tee -a "${MASTER_LOG}"
echo "║  Logs:                                          ║" | tee -a "${MASTER_LOG}"
echo "║    ${MASTER_LOG}  ║" | tee -a "${MASTER_LOG}"
echo "╚══════════════════════════════════════════════════╝" | tee -a "${MASTER_LOG}"

echo "" | tee -a "${MASTER_LOG}"
log "Output files:"
ls -lh "${OUTPUT_DIR}"/ 2>/dev/null | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"
log "Log files:"
ls -lh "${LOG_DIR}"/ 2>/dev/null | tee -a "${MASTER_LOG}"
