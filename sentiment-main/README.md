# Sentiment Analysis Pipeline

A high-performance sentiment analysis pipeline that integrates audio and text inputs using a frozen-cache strategy with **WavLM-Large** and **HuBERT-Large**.

## Features

- **Frozen-Cache Architecture**: Extracts and caches encoder features for WavLM and HuBERT to eliminate redundant forward passes during training.
- **Efficient Training**:
  - **4-5x speedup** over standard audio models by caching 80% of encoder layers.
  - **LoRA-optimized text encoder**: Adapts a pre-trained DeBERTa-v3-large for sentiment classification with minimal trainable parameters.
  - **Quantum-Inspired Fusion**:
    - **Variational Quantum Circuit (VQC)** for flexible fusion of audio and text features.
    - **Progressive Quantum Folding** reduces circuit depth while maintaining high expressivity.
- **Robust Preprocessing**:
  - **IEMOCAP Dataset**: Processes the standard IEMOCAP dataset with proper handling of multiple sessions and emotion labels.
  - **Audio Processing**: Automatic downsampling, padding, and truncation to `MAX_AUDIO_SEC`.
  - **Text Processing**: Tokenization with proper truncation and attention mask generation.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support for GPU acceleration)
- Hugging Face `transformers`
- `librosa`, `numpy`, `pandas`, `tqdm`
- CUDA-enabled GPU recommended for optimal performance (4090/A100/H100 tested).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rafat-Aman/sentiment-analysis-pipeline.git
   cd sentiment-analysis-pipeline/cached_pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Edit `config.py` to tune hyperparameters:

- **Model IDs**: Change `WAVLM_ID`, `HUBERT_ID`, `DEBERTA_ID` if needed.
- **Audio Settings**: Adjust `MAX_AUDIO_SEC`, `N_FROZEN`, `N_UNFROZEN`.
- **Quantum Settings**: Modify `PROJ_DIM`, `P_DEPTH`, `N_PQC_LAYERS`.
- **Training**: Tune `BATCH_SIZE`, `EPOCHS`, `LR`, `DROP`, etc.

## Usage

### 1. Pre-compute Frozen Cache

Run this script **once** before training to extract and cache WavLM and HuBERT encoder features. This step eliminates 80% of the encoder computation during training.

```bash
python build_frozen_cache.py [--max_audio_sec <seconds>] [--save_dir <path>] [--iemocap_path <path>]
```

- `--max_audio_sec`: Maximum audio duration per sample (default: 8s).
- `--save_dir`: Output directory for cache and parsed data (default: `./output`).
- `--iemocap_path`: Path to the IEMOCAP dataset (optional).

**Example**:
```bash
python build_frozen_cache.py --max_audio_sec 4 --save_dir ./output
```

This will create:
- `output/frozen_features.pt`: Cached WavLM and HuBERT hidden states.
- `output/iemocap_parsed.csv`: Parsed IEMOCAP metadata.

### 2. Train the Model

Run the training script using the frozen cache:

```bash
python train_cached.py [--epochs <num>] [--resume <path>] [--fold <id>] [--use_amp]
```

- `--epochs`: Number of training epochs (default: 15).
- `--resume`: Path to a previous checkpoint to resume training.
- `--fold`: Which 5-fold split to train (0-4, default: 0).
- `--use_amp`: Enable automatic mixed precision for faster training.

**Example**:
```bash
python train_cached.py --epochs 10 --fold 2 --use_amp
```

### 3. Evaluate the Model

Evaluate a trained model on the test set:

```bash
python evaluate.py --fold <id> [--model_path <path>]
```

- `--fold`: Which 5-fold split to evaluate (default: 0).
- `--model_path`: Path to a trained model checkpoint (optional, uses latest checkpoint if not specified).

### 4. Predict on New Data

Use the trained model to predict sentiment on new audio-text pairs:

```bash
python predict_cached.py --text "I'm feeling very happy today" --audio_path /path/to/audio.wav [--model_path <path>]
```

## Model Architecture

The model consists of three main components:

1. **Frozen Encoders (WavLM & HuBERT)**
   - Extract high-level acoustic features.
   - Only the last `N_UNFROZEN` layers are trainable; the rest are frozen and cached.

2. **Text Encoder (DeBERTa-v3-large with LoRA)**
   - Pre-trained DeBERTa encoder with **LoRA adapters** for efficient adaptation.
   - Adapts text to `CA_DIM` dimension.

3. **Quantum-Inspired Fusion Network**
   - **Attention Pooling**: Pools audio and text features.
   - **Quantum Fusion Layer (QFL)**:
     - **Multi-head attention** with quantum-inspired projections.
     - **Variational Quantum Circuit (VQC)** with parameter-efficient ansatz.
     - **Progressive Quantum Folding** for optimized circuit depth.
   - **Classification Head**: Maps fused features to emotion classes.

## Dataset Details

### IEMOCAP Dataset

The pipeline uses the [Interactive Emotional Dyadic Motion Capture (IEMOCAP)](https://sail.usc.edu/iemocap/) dataset, which contains:

- 5 sessions with dyadic interactions
- 10 speakers (5 male, 5 female)
- 6 primary emotion categories:
  -anger
  -happiness
  -excitement
  -neutral
  -sadness
  -surprise

**Emotion Mapping**:
```python
EMO_MAP = {
    'ang': 'anger',
    'hap': 'happy',
    'exc': 'happy',
    'neu': 'neutral',
    'sad': 'sadness',
    'fru': 'anger',
    'sur': 'happy',
    'oth': None,  # Excluded
}

CLASSES = ['anger', 'happy', 'neutral', 'sadness']
```

### 5-Fold Cross-Validation

The dataset is split into 5 folds based on **speakers** to prevent data leakage:

| Fold | Speakers                                   | Train Set               | Test Set |
|------|--------------------------------------------|-------------------------|----------|
| 0    | Speakers {1, 4, 6, 7, 8}                   | 514 samples             | 172 samples |
| 1    | Speakers {2, 5, 6, 7, 9}                   | 494 samples             | 172 samples |
| 2    | Speakers {1, 3, 6, 8, 9}                   | 507 samples             | 172 samples |
| 3    | Speakers {1, 2, 4, 5, 7}                   | 504 samples             | 172 samples |
| 4    | Speakers {2, 3, 4, 5, 8, 9} (larger)       | 609 samples             | 172 samples |

## Performance

### Achieved Metrics (Example)

The system is optimized for **F1-score** (macro-average).

**Example Test Performance**:

```
============================================================
TEST PERFORMANCE - FOLD 0
============================================================
                 precision    recall  f1-score   support
          anger       0.67      0.52      0.58        46
          happy       0.60      0.67      0.63        48
        neutral       0.75      0.74      0.74        46
        sadness       0.64      0.61      0.62        32

       accuracy                           0.64
