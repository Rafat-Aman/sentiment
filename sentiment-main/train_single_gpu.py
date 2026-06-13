# ============================================================
# train_single_gpu.py — Single-GPU training with frozen cache
# ============================================================
# 5-fold stratified CV using pre-cached frozen features.
# No model parallelism — everything on one GPU.
#
# Usage:
#   python train_single_gpu.py --cache_path ./output/frozen_features.pt
# ============================================================
import os, sys, argparse, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from config import *
from parse_iemocap import parse_iemocap, find_iemocap
from dataset_cached import IEMOCAPDatasetCached, collate_cached
from model_cached import QuantumEnhancedCachedModel

warnings.filterwarnings('ignore')


def class_weights(labels, n, dev):
    counts = np.bincount(labels, minlength=n).astype(float)
    w = counts.sum() / (n * counts)
    return torch.tensor(w, dtype=torch.float, device=dev)


def train_epoch(model, loader, optimizer, criterion, scaler, device,
                clip=CLIP_GRAD, accum=ACCUM_STEPS):
    model.train()
    total = 0.0
    optimizer.zero_grad()
    bar = tqdm(loader, desc='  train', leave=False)
    for step, b in enumerate(bar):
        wf  = b['wavlm_f']
        hf  = b['hubert_f']
        ids = b['input_ids']
        msk = b['attention_mask']
        lbl = b['labels'].to(device)

        if USE_AMP:
            with torch.amp.autocast('cuda'):
                loss = criterion(
                    model(wavlm_f=wf, hubert_f=hf,
                          input_ids=ids, attention_mask=msk), lbl) / accum
            scaler.scale(loss).backward()
        else:
            loss = criterion(
                model(wavlm_f=wf, hubert_f=hf,
                      input_ids=ids, attention_mask=msk), lbl) / accum
            loss.backward()

        is_last = (step + 1) == len(loader)
        if (step + 1) % accum == 0 or is_last:
            if USE_AMP:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
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
        wf  = b['wavlm_f']
        hf  = b['hubert_f']
        ids = b['input_ids']
        msk = b['attention_mask']
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                logits = model(wavlm_f=wf, hubert_f=hf,
                               input_ids=ids, attention_mask=msk)
        else:
            logits = model(wavlm_f=wf, hubert_f=hf,
                           input_ids=ids, attention_mask=msk)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', type=str, required=True,
                        help='Path to frozen_features.pt')
    parser.add_argument('--iemocap_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to iemocap_parsed.csv (skips re-parsing)')
    args = parser.parse_args()

    # Seed
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # Load cache
    print(f'Loading frozen cache: {args.cache_path}')
    cache = torch.load(args.cache_path, map_location='cpu', weights_only=True)
    print(f'  Entries: {len(cache)}')

    # Load DataFrame
    if args.csv_path and os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
        print(f'Loaded DataFrame from {args.csv_path}: {len(df)} samples')
    else:
        base_path = find_iemocap(args.iemocap_path)
        if not base_path:
            csv_fallback = os.path.join(args.save_dir, 'iemocap_parsed.csv')
            if os.path.exists(csv_fallback):
                df = pd.read_csv(csv_fallback)
                print(f'Loaded from {csv_fallback}: {len(df)} samples')
            else:
                print('ERROR: IEMOCAP not found and no CSV available.')
                sys.exit(1)
        else:
            df = parse_iemocap(base_path)

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 5-Fold CV ────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    all_true, all_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'])):
        print(f'\n{"="*60}')
        print(f'  FOLD {fold+1}/{N_FOLDS}  |  cached_qfl  |  single-GPU')
        print(f'{"="*60}')

        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        print(f'  train={len(df_tr)}  test={len(df_te)}')

        tr_loader = DataLoader(
            IEMOCAPDatasetCached(df_tr, cache), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_cached, num_workers=2, pin_memory=True)
        te_loader = DataLoader(
            IEMOCAPDatasetCached(df_te, cache), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_cached, num_workers=2, pin_memory=True)

        model = QuantumEnhancedCachedModel(NUM_CLASSES, device)

        cw = class_weights(df_tr['label'].values, NUM_CLASSES, device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

        best_f1 = 0.0
        ckpt_path = os.path.join(args.save_dir, f'cached_qfl_fold{fold+1}.pth')

        for ep in range(EPOCHS):
            print(f'\n  Epoch {ep+1}/{EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})')
            avg_loss = train_epoch(model, tr_loader, optimizer, criterion,
                                   scaler, device)
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
        best_model = QuantumEnhancedCachedModel(NUM_CLASSES, device)
        best_model.load_state_dict(
            torch.load(ckpt_path, map_location='cpu', weights_only=True))
        true_l, pred_l = evaluate(best_model, te_loader, device)
        fold_results.append(report(true_l, pred_l, CLASSES, prefix=f'  [Fold {fold+1}]'))
        all_true.extend(true_l.tolist())
        all_pred.extend(pred_l.tolist())

        del model, best_model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  RESULTS  |  cached_qfl  |  single-GPU')
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
