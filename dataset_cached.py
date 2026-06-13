# ============================================================
# dataset_cached.py — Dataset & collate for frozen-cache training
# ============================================================
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import DEBERTA_ID, MAX_TEXT_LEN

text_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_ID)


class IEMOCAPDatasetCached(Dataset):
    """Loads pre-computed frozen features instead of raw audio."""
    def __init__(self, dataframe, cache):
        self.df    = dataframe.reset_index(drop=True)
        self.cache = cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        text_enc = text_tokenizer(
            row['text'], padding='max_length', truncation=True,
            max_length=MAX_TEXT_LEN, return_tensors='pt')
        c = self.cache[row['utt_id']]
        return {
            'wavlm_f':        c['w'].float(),       # [T, 1024]
            'hubert_f':       c['h'].float(),       # [T, 1024]
            'input_ids':      text_enc.input_ids.squeeze(0),
            'attention_mask': text_enc.attention_mask.squeeze(0),
            'label':          torch.tensor(row['label'], dtype=torch.long),
        }


def collate_cached(batch):
    """Pad variable-length audio features to batch max."""
    ws = [b['wavlm_f']  for b in batch]
    hs = [b['hubert_f'] for b in batch]
    T  = max(w.shape[0] for w in ws)
    return {
        'wavlm_f':        torch.stack([F.pad(w, (0, 0, 0, T - w.shape[0])) for w in ws]),
        'hubert_f':       torch.stack([F.pad(h, (0, 0, 0, T - h.shape[0])) for h in hs]),
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask':  torch.stack([b['attention_mask'] for b in batch]),
        'labels':         torch.stack([b['label']          for b in batch]),
    }
