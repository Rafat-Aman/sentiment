# ============================================================
# config.py — Shared configuration
# ============================================================
# Full fine-tuning on 48GB RTX 4090.
# No caching — all encoders trained end-to-end.
# QFL hyperparameters increased for richer quantum interactions.
# ============================================================
import os, math

# QFL hyperparameters (INCREASED for 48GB)
PROJ_DIM     = 16    # ↑ from 8 — MD=32, doubles quantum state space (64-dim Hilbert)
P_DEPTH      = 1     # P=1 still optimal for <10k samples (avoids overfitting)
N_PQC_LAYERS = 8     # ↑ from 5 — more expressive hardware-efficient ansatz

# Architecture
CA_DIM  = 512        # Matched to original (keeps classifier width manageable)
HEADS   = 8
DROP    = 0.3        # Matched to original

# LoRA (text encoder)
LORA_R       = 64    # High rank for maximum text adaptation
LORA_ALPHA   = 256   # alpha/r = 4
LORA_DROPOUT = 0.1

# Training
BATCH_SIZE   = 8     # Full fine-tuning uses more VRAM
ACCUM_STEPS  = 4     # effective batch = 32
EPOCHS       = 10    # Original used 5; more time to converge with bigger model
LR           = 1e-5  # Matched to original
WEIGHT_DECAY = 0.01
CLIP_GRAD    = 1.0
N_FOLDS      = 5
LABEL_SMOOTH = 0.0   # Matched to original (no smoothing)
USE_AMP      = False  # QFL produces NaN under FP16

# Audio
MAX_TEXT_LEN  = 64
MAX_AUDIO_SEC = 8
SAMPLE_RATE   = 16000

# Model IDs (upgraded from wav2vec2-base / roberta-base)
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
MD       = 2 * PROJ_DIM   # 32
N_INDEX  = max(1, math.ceil(math.log2(MD + 1)))  # 5
N_QUBITS = N_INDEX + 1    # 6 → state_dim = 64

SEED = 42
