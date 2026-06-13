# ============================================================
# parse_iemocap.py — IEMOCAP dataset parser
# ============================================================
import os, re
import pandas as pd
from config import EMO_MAP, LABEL2IDX, CLASSES


def parse_iemocap(base_path):
    """Parse IEMOCAP directory → DataFrame."""
    records, dropped = [], 0
    for session in range(1, 6):
        eval_dir = os.path.join(base_path, f'Session{session}', 'dialog', 'EmoEvaluation')
        text_dir = os.path.join(base_path, f'Session{session}', 'dialog', 'transcriptions')
        wav_base = os.path.join(base_path, f'Session{session}', 'sentences', 'wav')
        if not os.path.exists(eval_dir):
            print(f'[!] Session {session} not found'); continue
        for fname in sorted(os.listdir(eval_dir)):
            if not fname.endswith('.txt'): continue
            emo_dict = {}
            with open(os.path.join(eval_dir, fname)) as f:
                for line in f:
                    if not line.startswith('['): continue
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    uid, raw = parts[1].strip(), parts[2].strip()
                    if raw in EMO_MAP:
                        emo_dict[uid] = LABEL2IDX[EMO_MAP[raw]]
                    else:
                        dropped += 1
            tpath = os.path.join(text_dir, fname)
            if not os.path.exists(tpath): continue
            with open(tpath) as f:
                for line in f:
                    m = re.match(r'^(\w+)\s+\[.+\]:\s+(.+)$', line.strip())
                    if not m: continue
                    uid, text = m.group(1), m.group(2).strip()
                    if uid not in emo_dict: continue
                    wav_path = os.path.join(
                        wav_base, '_'.join(uid.split('_')[:-1]), f'{uid}.wav')
                    if os.path.exists(wav_path):
                        records.append({
                            'utt_id': uid, 'text': text,
                            'file_path': wav_path,
                            'label': emo_dict[uid], 'session': session,
                        })
    df = pd.DataFrame(records)
    print(f'Total samples : {len(df)} (dropped {dropped} unsupported labels)')
    for i, c in enumerate(CLASSES):
        n = (df['label'] == i).sum()
        print(f'  {c}: {n} ({100 * n / len(df):.1f}%)')
    return df


def find_iemocap(user_path=None):
    """Auto-detect IEMOCAP path."""
    if user_path and os.path.exists(user_path):
        return user_path
    possible = [
        '/workspace/data/IEMOCAP_full_release',
        '/kaggle/input/datasets/dejolilandry/iemocapfullrelease/IEMOCAP_full_release',
        '/kaggle/input/iemocap-full-release/IEMOCAP_full_release',
        './IEMOCAP_full_release',
        './data/IEMOCAP_full_release',
    ]
    return next((p for p in possible if os.path.exists(p)), None)
