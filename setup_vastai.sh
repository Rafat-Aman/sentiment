#!/bin/bash
# ============================================================
# setup_vastai.sh — Full environment setup for vast.ai GPU instance
# ============================================================
# Installs all dependencies, downloads datasets, and prepares
# the environment for frozen-cache training.
#
# Usage:
#   chmod +x setup_vastai.sh && ./setup_vastai.sh
# ============================================================
set -e

WORK_DIR="/workspace"
DATA_DIR="${WORK_DIR}/data"
CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="${WORK_DIR}/output"

echo "============================================"
echo "  VAST.AI SETUP — Quantum Enhanced Fusion"
echo "============================================"
echo "  Work dir: ${WORK_DIR}"
echo ""

# ─────────────────────────────────────────────────────────────
# 1. SYSTEM PACKAGES
# ─────────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq \
    ffmpeg libsndfile1 git wget unzip p7zip-full > /dev/null 2>&1
echo "  ✅ System packages installed"

# ─────────────────────────────────────────────────────────────
# 2. PYTHON DEPENDENCIES
# ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip

if python -c "import torch" 2>/dev/null; then
    echo "  ✅ Pre-installed PyTorch detected. Skipping torch/torchvision/torchaudio install."
    pip install \
        transformers \
        peft \
        librosa \
        soundfile \
        pandas \
        numpy \
        scikit-learn \
        tqdm \
        kagglehub
else
    echo "  Installing PyTorch and other dependencies..."
    pip install \
        torch torchvision torchaudio \
        transformers \
        peft \
        librosa \
        soundfile \
        pandas \
        numpy \
        scikit-learn \
        tqdm \
        kagglehub
fi

echo "  ✅ Python dependencies installed"

# Verify key imports
echo "  Verifying imports..."
python -c "
import torch, transformers, peft, librosa, sklearn
print(f'    PyTorch: {torch.__version__}')
print(f'    Transformers: {transformers.__version__}')
print(f'    CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    GPU: {torch.cuda.get_device_name(0)}')
    print(f'    VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ─────────────────────────────────────────────────────────────
# 3. DOWNLOAD DATASETS
# ─────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Downloading datasets..."
mkdir -p "${DATA_DIR}"

# IEMOCAP via kagglehub
echo "  Downloading IEMOCAP..."
python -c "
import kagglehub, os, shutil
path = kagglehub.dataset_download('dejolilandry/iemocapfullrelease')
print(f'    Downloaded to: {path}')
# Find IEMOCAP_full_release directory
for root, dirs, files in os.walk(path):
    if 'IEMOCAP_full_release' in dirs:
        src = os.path.join(root, 'IEMOCAP_full_release')
        dst = '${DATA_DIR}/IEMOCAP_full_release'
        if not os.path.exists(dst):
            os.symlink(src, dst)
            print(f'    Linked: {dst}')
        else:
            print(f'    Already exists: {dst}')
        break
else:
    # Direct link if structure is flat
    dst = '${DATA_DIR}/IEMOCAP_full_release'
    if not os.path.exists(dst):
        os.symlink(path, dst)
        print(f'    Linked: {dst}')
"
echo "  ✅ IEMOCAP ready"

# MELD via kagglehub
echo "  Downloading MELD..."
python -c "
import kagglehub, os
path = kagglehub.dataset_download('zaber666/meld-dataset')
print(f'    Downloaded to: {path}')
dst = '${DATA_DIR}/MELD'
if not os.path.exists(dst):
    os.symlink(path, dst)
    print(f'    Linked: {dst}')
else:
    print(f'    Already exists: {dst}')
"
echo "  ✅ MELD ready"

# ─────────────────────────────────────────────────────────────
# 4. COPY PIPELINE CODE
# ─────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Setting up pipeline code..."
mkdir -p "${CODE_DIR}"
mkdir -p "${OUTPUT_DIR}"

# If code isn't already in place, note for user
if [ ! -f "${CODE_DIR}/config.py" ]; then
    echo "  ⚠️  Pipeline code not found at ${CODE_DIR}"
    echo "  Upload your cached_pipeline/ directory to ${CODE_DIR}"
    echo "  Files needed:"
    echo "    - config.py"
    echo "    - parse_iemocap.py"
    echo "    - build_frozen_cache.py"
    echo "    - dataset_cached.py"
    echo "    - qfl.py"
    echo "    - model_cached.py"
    echo "    - train_single_gpu.py"
else
    echo "  ✅ Pipeline code found"
fi

# ─────────────────────────────────────────────────────────────
# 5. UPDATE CONFIG PATHS
# ─────────────────────────────────────────────────────────────
echo ""
echo "[5/6] Verifying dataset paths..."

# Check IEMOCAP
if [ -d "${DATA_DIR}/IEMOCAP_full_release" ]; then
    echo "  ✅ IEMOCAP: ${DATA_DIR}/IEMOCAP_full_release"
    # Count sessions
    SESSIONS=$(ls -d ${DATA_DIR}/IEMOCAP_full_release/Session* 2>/dev/null | wc -l)
    echo "     Sessions found: ${SESSIONS}"
else
    echo "  ❌ IEMOCAP not found at ${DATA_DIR}/IEMOCAP_full_release"
fi

# Check MELD
if [ -d "${DATA_DIR}/MELD" ]; then
    echo "  ✅ MELD: ${DATA_DIR}/MELD"
else
    echo "  ❌ MELD not found at ${DATA_DIR}/MELD"
fi

# ─────────────────────────────────────────────────────────────
# 6. PRE-DOWNLOAD HF MODELS (avoids timeout during training)
# ─────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Pre-downloading HuggingFace models (this takes a while)..."
python -c "
from transformers import WavLMModel, HubertModel, DebertaV2Model
from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer

print('  Downloading WavLM-Large...')
WavLMModel.from_pretrained('microsoft/wavlm-large')
Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')

print('  Downloading HuBERT-Large...')
HubertModel.from_pretrained('facebook/hubert-large-ls960-ft')

print('  Downloading DeBERTa-v3-Large...')
DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')
AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

print('  ✅ All models cached locally')
"

# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  ✅ SETUP COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Clone repository to ${CODE_DIR}/ (if not done)"
echo ""
echo "  2. Build frozen cache (run once, ~30 min):"
echo "     cd ${CODE_DIR}"
echo "     python build_frozen_cache.py \\"
echo "       --iemocap_path ${DATA_DIR}/IEMOCAP_full_release \\"
echo "       --save_dir ${OUTPUT_DIR}"
echo ""
echo "  3. Train (runs 5-fold CV):"
echo "     cd ${CODE_DIR}"
echo "     python train_single_gpu.py \\"
echo "       --cache_path ${OUTPUT_DIR}/frozen_features.pt \\"
echo "       --iemocap_path ${DATA_DIR}/IEMOCAP_full_release \\"
echo "       --save_dir ${OUTPUT_DIR}"
echo ""
echo "  Estimated training time:"
echo "    T4:   ~1.5 hours"
echo "    4090: ~1 hour"
echo "    A100: ~1 hour"
echo "    H100: ~35 min"
echo ""
