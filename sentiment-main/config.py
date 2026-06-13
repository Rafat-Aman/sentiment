# ============================================================
# config.py — Shared configuration for frozen-cache pipeline
# ============================================================
# Tuned for high-VRAM GPUs (4090/A100/H100 on vast.ai).
# Prioritises F1 score over VRAM frugality.
# ============================================================
import os, math

# QFL hyperparameters
PROJ_DIM     = 32    # ↑ from 16 — richer quantum state space
P_DEPTH      = 1     # P=1 still optimal for <10k samples
N_PQC_LAYERS = 12    # ↑ from 10 — more ansatz expressivity

# Architecture
CA_DIM  = 1024       # cross-attention hidden dim
HEADS   = 8
DROP    = 0.25       # ↓ from 0.3 — less regularisation with larger batches

# LoRA (text encoder)
LORA_R       = 32    # ↑ from 16 — doubles LoRA capacity
LORA_ALPHA   = 128   # ↑ from 64 — scales with rank (alpha/r = 4)
LORA_DROPOUT = 0.1

# Training
BATCH_SIZE   = 16    # ↑ from 4/8 — 4090 has headroom, A100/H100 easily
ACCUM_STEPS  = 2     # effective batch = 32 (was 16)
EPOCHS       = 15    # ↑ from 10 — more time to converge with richer model
LR           = 1e-5
WEIGHT_DECAY = 0.01
CLIP_GRAD    = 1.0
N_FOLDS      = 5
LABEL_SMOOTH = 0.1
USE_AMP      = True

# Audio
MAX_TEXT_LEN  = 64
MAX_AUDIO_SEC = 8    # ↑ back to 8 — full context matters for emotion
SAMPLE_RATE   = 16000
N_FROZEN      = 16   # ↓ from 20 — unfreeze 8 layers instead of 4
                      # more fine-tuning capacity = better F1
N_UNFROZEN    = 8    # 24 total encoder layers - 16 frozen = 8 unfrozen

# Model IDs
WAVLM_ID   = 'microsoft/wavlm-large'
HUBERT_ID  = 'facebook/hubert-large-ls960-ft'
DEBERTA_ID = 'microsoft/deberta-v3-large'

# Class map (kawaoto)
EMO_MAP = {
    'ang': 'anger', 'hap': 'happy', 'exc': 'happy',
    'neu': 'neutral', 'sad': 'sadness',
}
CLASSES     = ['anger', 'happy', 'neutral', 'sadness']
LABEL2IDX   = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Derived QFL dims
MD       = 2 * PROJ_DIM
N_INDEX  = max(1, math.ceil(math.log2(MD + 1)))
N_QUBITS = N_INDEX + 1

SEED = 42
