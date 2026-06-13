# ============================================================
# model_full.py — Full fine-tuning model (no caching)
# ============================================================
# All encoders trained end-to-end on a single 48GB GPU.
# WavLM-Large + HuBERT-Large fully fine-tuned (feature extractor frozen).
# DeBERTa-v3-Large with high-rank LoRA.
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, HubertModel, DebertaV2Model
from peft import LoraConfig, get_peft_model, TaskType

from config import (
    WAVLM_ID, HUBERT_ID, DEBERTA_ID,
    CA_DIM, HEADS, DROP, PROJ_DIM, P_DEPTH, N_PQC_LAYERS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
)
from qfl import QuantumFusionLayer


def _proj(in_dim, out_dim, drop):
    """Lightweight 2-layer projection (matched to original _proj2)."""
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.LayerNorm(256),
        nn.ReLU(), nn.Dropout(drop),
        nn.Linear(256, out_dim),
    )


class QuantumEnhancedFullModel(nn.Module):
    """
    Full fine-tuning model: WavLM-Large + HuBERT-Large + DeBERTa-v3-Large.
    All audio encoder layers are trainable (only feature extractors frozen).
    Text encoder uses high-rank LoRA for parameter-efficient adaptation.

    Architecture mirrors the original QuantumEnhancedModel but with
    dual audio encoders and upgraded backbone models.
    """

    def __init__(self, num_classes, device,
                 ca_dim=CA_DIM, proj_dim=PROJ_DIM,
                 P=P_DEPTH, layers=N_PQC_LAYERS,
                 heads=HEADS, drop=DROP):
        super().__init__()
        self.device = device

        # ── WavLM-Large: full fine-tuning ─────────────────────
        self.wavlm = WavLMModel.from_pretrained(WAVLM_ID).to(device)
        self.wavlm.feature_extractor._freeze_parameters()
        self.wavlm.gradient_checkpointing_enable()
        ae_dim = self.wavlm.config.hidden_size  # 1024

        # ── HuBERT-Large: full fine-tuning ────────────────────
        self.hubert = HubertModel.from_pretrained(HUBERT_ID).to(device)
        self.hubert.feature_extractor._freeze_parameters()
        self.hubert.gradient_checkpointing_enable()
        he_dim = self.hubert.config.hidden_size  # 1024

        # ── DeBERTa-v3-Large (LoRA) ──────────────────────────
        base_te = DebertaV2Model.from_pretrained(DEBERTA_ID)
        base_te.gradient_checkpointing_enable()
        for p in base_te.parameters():
            p.requires_grad = False
        te_dim = base_te.config.hidden_size  # 1024
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=['query_proj', 'value_proj', 'key_proj'],
            bias='none', task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.te = get_peft_model(base_te, lora_cfg).to(device)

        # ── Cross-attention path ─────────────────────────────
        self.ap = nn.Linear(ae_dim, ca_dim).to(device)
        self.tp = nn.Linear(te_dim, ca_dim).to(device)
        self.ca = nn.MultiheadAttention(
            ca_dim, heads, batch_first=True, dropout=drop).to(device)

        # ── QFL path ─────────────────────────────────────────
        self.qfl_ap_w = _proj(ae_dim, proj_dim, drop).to(device)
        self.qfl_ap_h = _proj(he_dim, proj_dim, drop).to(device)
        self.qfl_tp   = _proj(te_dim, proj_dim, drop).to(device)
        self.qfl = QuantumFusionLayer(
            MD=2 * proj_dim, P=P, n_pqc_layers=layers).to(device)

        # ── Classifier (matched to original structure) ───────
        fuse_dim = ca_dim * 3 + self.qfl.out_dim
        self.clf = nn.Sequential(
            nn.Linear(fuse_dim, 256), nn.LayerNorm(256),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, num_classes),
        ).to(device)

        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[FullModel] device={device}')
        print(f'  Total params    : {total/1e6:.2f}M')
        print(f'  Trainable params: {trainable/1e6:.2f}M')
        print(f'  QFL: MD={2*proj_dim}, out_dim={self.qfl.out_dim}')

    def forward(self, input_values, input_ids, attention_mask):
        dev = self.device

        # ── Audio encoders (full forward pass) ────────────────
        ha_w = self.wavlm(input_values.to(dev)).last_hidden_state.mean(dim=1).float()
        ha_h = self.hubert(input_values.to(dev)).last_hidden_state.mean(dim=1).float()

        # ── Text encoder ──────────────────────────────────────
        ht = self.te(
            input_ids.to(dev),
            attention_mask=attention_mask.to(dev),
        ).last_hidden_state[:, 0, :].float()

        # ── Cross-attention fusion ────────────────────────────
        ha_avg = (ha_w + ha_h) / 2
        ha_ca  = F.relu(self.ap(ha_avg))
        ht_ca  = F.relu(self.tp(ht))
        attn, _ = self.ca(
            ht_ca.unsqueeze(1), ha_ca.unsqueeze(1), ha_ca.unsqueeze(1))
        ca_feat = torch.cat([ht_ca, ha_ca, attn.squeeze(1)], dim=1)

        # ── QFL path ──────────────────────────────────────────
        qfl_audio = self.qfl_ap_w(ha_w) + self.qfl_ap_h(ha_h)
        qfl_text  = self.qfl_tp(ht)
        qfl_feat  = self.qfl(torch.cat([qfl_audio, qfl_text], dim=1))

        return self.clf(torch.cat([ca_feat, qfl_feat], dim=1))
