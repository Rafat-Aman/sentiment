# ============================================================
# build_frozen_cache.py — Pre-compute frozen encoder features
# ============================================================
# Run ONCE before training. Extracts WavLM + HuBERT hidden states
# at layer N_FROZEN, saves as FP16 in frozen_features.pt.
#
# Usage:
#   python build_frozen_cache.py [--iemocap_path /path] [--save_dir ./output]
# ============================================================
import os, sys, argparse, warnings
import torch
import librosa
from transformers import WavLMModel, HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

from config import WAVLM_ID, HUBERT_ID, SAMPLE_RATE, N_FROZEN, MAX_AUDIO_SEC
from parse_iemocap import parse_iemocap, find_iemocap

warnings.filterwarnings('ignore')


@torch.no_grad()
def build_frozen_cache(df, wavlm, hubert, processor, max_audio_sec, device):
    cache = {}
    wavlm.eval(); hubert.eval()
    max_samples = max_audio_sec * SAMPLE_RATE

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Building cache'):
        wav, _ = librosa.load(row['file_path'], sr=SAMPLE_RATE)
        wav = wav[:max_samples]
        iv = processor(wav, sampling_rate=SAMPLE_RATE,
                       return_tensors='pt').input_values.to(device)

        w_h = wavlm(iv, output_hidden_states=True).hidden_states[N_FROZEN]
        h_h = hubert(iv, output_hidden_states=True).hidden_states[N_FROZEN]

        cache[row['utt_id']] = {
            'w': w_h.squeeze(0).cpu().half(),
            'h': h_h.squeeze(0).cpu().half(),
        }
        del iv, w_h, h_h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cache


def main():
    parser = argparse.ArgumentParser(description='Pre-compute frozen features')
    parser.add_argument('--iemocap_path', type=str, default=None)
    parser.add_argument('--max_audio_sec', type=int, default=MAX_AUDIO_SEC)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    base_path = find_iemocap(args.iemocap_path)
    if not base_path:
        print('ERROR: IEMOCAP not found. Use --iemocap_path'); sys.exit(1)
    print(f'IEMOCAP: {base_path}')

    df = parse_iemocap(base_path)
    os.makedirs(args.save_dir, exist_ok=True)
    cache_path = os.path.join(args.save_dir, 'frozen_features.pt')

    if os.path.exists(cache_path):
        existing = torch.load(cache_path, map_location='cpu', weights_only=True)
        if len(existing) == len(df):
            print(f'Cache already complete ({len(existing)} entries). Skipping.')
            return
        del existing

    print(f'Loading WavLM-Large...'); wavlm = WavLMModel.from_pretrained(WAVLM_ID).to(device)
    print(f'Loading HuBERT-Large...'); hubert = HubertModel.from_pretrained(HUBERT_ID).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_ID)

    print(f'\nCaching layers 1-{N_FROZEN}, max_audio={args.max_audio_sec}s\n')
    cache = build_frozen_cache(df, wavlm, hubert, processor, args.max_audio_sec, device)

    torch.save(cache, cache_path)
    print(f'\n✅ Cache saved: {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)')

    first_key = next(iter(cache))
    print(f'   Sample: w={cache[first_key]["w"].shape}, h={cache[first_key]["h"].shape}')

    df.to_csv(os.path.join(args.save_dir, 'iemocap_parsed.csv'), index=False)
    del wavlm, hubert, cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
