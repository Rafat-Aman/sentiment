# ============================================================
# train_full.py — Full fine-tuning training (no caching)
# ============================================================
# 5-fold stratified CV with end-to-end training.
# All audio encoders fully fine-tuned on single 48GB GPU.
#
# Usage:
#   python train_full.py --iemocap_path /path/to/IEMOCAP_full_release
# ============================================================
import os, sys, argparse, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer
from tqdm import tqdm

from config import *
from parse_iemocap import parse_iemocap, find_iemocap
from model_full import QuantumEnhancedFullModel

warnings.filterwarnings('ignore')


# ── Dataset ──────────────────────────────────────────────────

audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_ID)
text_tokenizer  = AutoTokenizer.from_pretrained(DEBERTA_ID)


class IEMOCAPDataset(Dataset):
    """Raw audio dataset (no caching)."""
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                row = self.df.iloc[idx]
                text_enc = text_tokenizer(
                    row['text'], padding='max_length', truncation=True,
                    max_length=MAX_TEXT_LEN, return_tensors='pt')
                wav, _ = librosa.load(row['file_path'], sr=SAMPLE_RATE)
                wav = wav[:MAX_AUDIO_SEC * SAMPLE_RATE]
                aud_enc = audio_processor(
                    wav, sampling_rate=SAMPLE_RATE, return_tensors='pt')
                return {
                    'input_values':   aud_enc.input_values.squeeze(0),
                    'input_ids':      text_enc.input_ids.squeeze(0),
                    'attention_mask': text_enc.attention_mask.squeeze(0),
                    'label': torch.tensor(row['label'], dtype=torch.long),
                }
            except Exception as e:
                print(f'[Dataset] attempt {attempt+1} failed idx={idx}: {e}')
                idx = random.randint(0, len(self.df) - 1)
        raise RuntimeError('Too many failed samples.')


def collate_fn(batch):
    """Pad variable-length audio to batch max."""
    input_ids      = torch.stack([b['input_ids']      for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels         = torch.stack([b['label']          for b in batch])
    audios  = [b['input_values'] for b in batch]
    max_len = max(a.shape[0] for a in audios)
    audio_padded = torch.stack(
        [F.pad(a, (0, max_len - a.shape[0])) for a in audios])
    return {
        'input_values':  audio_padded,
        'input_ids':     input_ids,
        'attention_mask': attention_mask,
        'labels':        labels,
    }


# ── Training ─────────────────────────────────────────────────

def class_weights(labels, n, dev):
    counts = np.bincount(labels, minlength=n).astype(float)
    w = counts.sum() / (n * counts)
    return torch.tensor(w, dtype=torch.float, device=dev)


def train_epoch(model, loader, optimizer, criterion, device,
                clip=CLIP_GRAD, accum=ACCUM_STEPS):
    model.train()
    total = 0.0
    optimizer.zero_grad()
    bar = tqdm(loader, desc='  train', leave=False)
    for step, b in enumerate(bar):
        iv  = b['input_values']
        ids = b['input_ids']
        msk = b['attention_mask']
        lbl = b['labels'].to(device)

        loss = criterion(
            model(input_values=iv, input_ids=ids, attention_mask=msk),
            lbl) / accum
        loss.backward()

        is_last = (step + 1) == len(loader)
        if (step + 1) % accum == 0 or is_last:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        total += loss.item() * accum
        bar.set_postfix(loss=f'{loss.item() * accum:.4f}')
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for b in tqdm(loader, desc='  eval ', leave=False):
        iv  = b['input_values']
        ids = b['input_ids']
        msk = b['attention_mask']
        logits = model(input_values=iv, input_ids=ids, attention_mask=msk)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(b['labels'].numpy())
    return np.array(labels), np.array(preds)


def report(true, pred, names, prefix=''):
    ua  = accuracy_score(true, pred)
    mf1 = f1_score(true, pred, average='macro',    zero_division=0)
    wf1 = f1_score(true, pred, average='weighted', zero_division=0)
    print(f'{prefix}  UA={ua:.4f}  MacroF1={mf1:.4f}  WtdF1={wf1:.4f}')
    print(classification_report(true, pred, target_names=names, zero_division=0))
    return {'ua': ua, 'macro_f1': mf1, 'weighted_f1': wf1}


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iemocap_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./output')
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  VRAM: {vram:.1f} GB')

    # Load DataFrame
    base_path = find_iemocap(args.iemocap_path)
    if not base_path:
        csv_fallback = os.path.join(args.save_dir, 'iemocap_parsed.csv')
        if os.path.exists(csv_fallback):
            df = pd.read_csv(csv_fallback)
        else:
            print('ERROR: IEMOCAP not found.'); sys.exit(1)
    else:
        df = parse_iemocap(base_path)

    print(f'Total samples : {len(df)}')
    for c in CLASSES:
        n = (df['label'] == LABEL2IDX[c]).sum()
        print(f'  {c}: {n} ({100*n/len(df):.1f}%)')

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 5-Fold CV ────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    all_true, all_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'])):
        print(f'\n{"="*60}')
        print(f'  FOLD {fold+1}/{N_FOLDS}  |  full_finetune  |  single-GPU')
        print(f'{"="*60}')

        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        print(f'  train={len(df_tr)}  test={len(df_te)}')

        tr_loader = DataLoader(
            IEMOCAPDataset(df_tr), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        te_loader = DataLoader(
            IEMOCAPDataset(df_te), batch_size=4,
            shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

        model = QuantumEnhancedFullModel(NUM_CLASSES, device)

        cw = class_weights(df_tr['label'].values, NUM_CLASSES, device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        print(f'  class_weights = {cw.cpu().numpy().round(3)}')

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_f1 = 0.0
        ckpt_path = os.path.join(args.save_dir, f'full_qfl_fold{fold+1}.pth')

        for ep in range(EPOCHS):
            print(f'\n  Epoch {ep+1}/{EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})')
            avg_loss = train_epoch(model, tr_loader, optimizer, criterion, device)
            scheduler.step()

            true_l, pred_l = evaluate(model, te_loader, device)
            ep_ua  = accuracy_score(true_l, pred_l)
            ep_mf1 = f1_score(true_l, pred_l, average='macro', zero_division=0)
            print(f'  loss={avg_loss:.4f}  UA={ep_ua:.4f}  MacroF1={ep_mf1:.4f}')

            if ep_mf1 > best_f1:
                best_f1 = ep_mf1
                torch.save(model.state_dict(), ckpt_path)
                print(f'  * New best MacroF1={best_f1:.4f} -> saved')

        # Final eval with best checkpoint
        best_model = QuantumEnhancedFullModel(NUM_CLASSES, device)
        best_model.load_state_dict(
            torch.load(ckpt_path, map_location='cpu', weights_only=True))
        true_l, pred_l = evaluate(best_model, te_loader, device)
        fold_results.append(report(true_l, pred_l, CLASSES, prefix=f'  [Fold {fold+1}]'))
        all_true.extend(true_l.tolist())
        all_pred.extend(pred_l.tolist())

        del model, best_model, optimizer, scheduler
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  RESULTS  |  full_finetune  |  single-GPU')
    print(f'{"="*60}')

    print(f'\n{"Fold":>5}  {"UA":>8}  {"MacroF1":>10}  {"WtdF1":>10}')
    print('-' * 42)
    for i, r in enumerate(fold_results):
        print(f'{i+1:>5}  {r["ua"]:>8.4f}  {r["macro_f1"]:>10.4f}  {r["weighted_f1"]:>10.4f}')
    print('-' * 42)

    avg_ua  = np.mean([r['ua']       for r in fold_results])
    avg_mf1 = np.mean([r['macro_f1'] for r in fold_results])
    avg_wf1 = np.mean([r['weighted_f1'] for r in fold_results])
    print(f'{"Mean":>5}  {avg_ua:>8.4f}  {avg_mf1:>10.4f}  {avg_wf1:>10.4f}')

    BASELINE_UA, BASELINE_MF1 = 0.7906, 0.7955
    print(f'\n  Baseline  UA={BASELINE_UA:.4f}  MacroF1={BASELINE_MF1:.4f}')
    print(f'  This run  UA={avg_ua:.4f}  MacroF1={avg_mf1:.4f}')
    print(f'  Delta UA      : {avg_ua - BASELINE_UA:+.4f}')
    print(f'  Delta MacroF1 : {avg_mf1 - BASELINE_MF1:+.4f}')

    report(np.array(all_true), np.array(all_pred), CLASSES, prefix='  [Pooled]')
    print('\n✅ Training complete.')


if __name__ == '__main__':
    main()
