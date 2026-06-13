# ============================================================
# model_cached.py — Single-GPU model using frozen cache
# ============================================================
# Only runs the LAST N_UNFROZEN layers of WavLM/HuBERT per step.
# Frozen layers 1..N_FROZEN are pre-computed in build_frozen_cache.py.
# No model parallelism needed — fits on a single GPU.
# Tuned for high-VRAM GPUs (4090/A100/H100).
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, HubertModel, DebertaV2Model
from peft import LoraConfig, get_peft_model, TaskType

from config import (
    WAVLM_ID, HUBERT_ID, DEBERTA_ID,
    CA_DIM, HEADS, DROP, PROJ_DIM, P_DEPTH, N_PQC_LAYERS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, N_UNFROZEN,
)
from qfl import QuantumFusionLayer


class QuantumEnhancedCachedModel(nn.Module):
    """
    Single-GPU model that consumes pre-cached frozen features.

    Forward signature:
        forward(wavlm_f, hubert_f, input_ids, attention_mask)
    where wavlm_f/hubert_f are [B, T, 1024] from the frozen cache.
    Only the last N_UNFROZEN encoder layers run per step.
    """

    def __init__(self, num_classes, device,
                 ca_dim=CA_DIM, proj_dim=PROJ_DIM,
                 P=P_DEPTH, layers=N_PQC_LAYERS,
                 heads=HEADS, drop=DROP):
        super().__init__()
        self.device = device

        # ── WavLM: only keep unfrozen layers (last N_UNFROZEN) ─
        _ae_full = WavLMModel.from_pretrained(WAVLM_ID)
        self.ae_unfrozen = nn.ModuleList(
            _ae_full.encoder.layers[-N_UNFROZEN:]
        ).to(device)
        ae_dim = _ae_full.config.hidden_size  # 1024
        # Keep layer norm if present
        self.ae_layer_norm = None
        if hasattr(_ae_full, 'layer_norm') and _ae_full.layer_norm is not None:
            self.ae_layer_norm = _ae_full.layer_norm.to(device)
        # WavLM: only layer 0 has rel_attn_embed for position bias.
        # We need it to pre-compute bias before running unfrozen layers.
        layer0_attn = _ae_full.encoder.layers[0].attention
        self.ae_rel_attn_embed = layer0_attn.rel_attn_embed.to(device)
        self.ae_num_buckets = layer0_attn.num_buckets
        self.ae_max_distance = layer0_attn.max_distance
        self.ae_num_heads = _ae_full.config.num_attention_heads
        del _ae_full

        # ── HuBERT: only keep unfrozen layers (last N_UNFROZEN) ─
        _he_full = HubertModel.from_pretrained(HUBERT_ID)
        self.he_unfrozen = nn.ModuleList(
            _he_full.encoder.layers[-N_UNFROZEN:]
        ).to(device)
        he_dim = _he_full.config.hidden_size  # 1024
        self.he_layer_norm = None
        if hasattr(_he_full, 'layer_norm') and _he_full.layer_norm is not None:
            self.he_layer_norm = _he_full.layer_norm.to(device)
        del _he_full

        # ── Attention pooling ────────────────────────────────
        self.pool_w = nn.Linear(ae_dim, 1).to(device)
        self.pool_h = nn.Linear(he_dim, 1).to(device)

        # ── DeBERTa-v3-Large (LoRA) ─────────────────────────
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

        # ── Cross-attention ──────────────────────────────────
        self.ap = nn.Linear(ae_dim, ca_dim).to(device)
        self.tp = nn.Linear(te_dim, ca_dim).to(device)
        self.ca = nn.MultiheadAttention(
            ca_dim, heads, batch_first=True, dropout=drop).to(device)

        # ── QFL path ─────────────────────────────────────────
        def _proj(in_d):
            return nn.Sequential(
                nn.Linear(in_d, 256), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(256, proj_dim),
            ).to(device)

        self.qfl_ap_w = _proj(ae_dim)
        self.qfl_ap_h = _proj(he_dim)
        self.qfl_tp   = _proj(te_dim)
        self.qfl = QuantumFusionLayer(
            MD=2 * proj_dim, P=P, n_pqc_layers=layers).to(device)

        fuse_dim = ca_dim * 3 + self.qfl.out_dim
        self.clf = nn.Sequential(
            nn.Linear(fuse_dim, 512), nn.LayerNorm(512),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 256), nn.LayerNorm(256),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, num_classes),
        ).to(device)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[CachedModel] device={device}, trainable={trainable/1e6:.2f}M')
        print(f'[QFL] MD={2*proj_dim}, out_dim={self.qfl.out_dim}')

    @staticmethod
    def _attn_pool(hidden_states, pool_linear):
        scores  = pool_linear(hidden_states).squeeze(-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)

    def _compute_wavlm_position_bias(self, seq_len, bsz):
        """Pre-compute WavLM relative position bias using layer-0's embedding."""
        import math
        context = torch.arange(seq_len, dtype=torch.long)[:, None]
        memory  = torch.arange(seq_len, dtype=torch.long)[None, :]
        relative = memory - context

        num_buckets = self.ae_num_buckets
        half = num_buckets // 2
        buckets = (relative > 0).long() * half
        rel_abs = relative.abs()
        max_exact = half // 2
        is_small = rel_abs < max_exact

        large = torch.log(rel_abs.float() / max_exact)
        large = large / math.log(self.ae_max_distance / max_exact)
        large = (max_exact + large * (half - max_exact)).long()
        large = torch.min(large, torch.full_like(large, half - 1))
        buckets += torch.where(is_small, rel_abs, large)

        buckets = buckets.to(self.ae_rel_attn_embed.weight.device)
        values = self.ae_rel_attn_embed(buckets).permute(2, 0, 1)  # [H, T, T]
        return values.unsqueeze(0).expand(bsz, -1, -1, -1) \
                     .reshape(bsz * self.ae_num_heads, seq_len, seq_len)

    def forward(self, wavlm_f, hubert_f, input_ids, attention_mask):
        dev = self.device

        # ── Run unfrozen WavLM layers ────────────────────────
        x_w = wavlm_f.to(dev)
        # Pre-compute position bias (only layer 0 has rel_attn_embed)
        position_bias = self._compute_wavlm_position_bias(x_w.size(1), x_w.size(0))
        for layer in self.ae_unfrozen:
            x_w, position_bias = layer(x_w, attention_mask=None,
                                       position_bias=position_bias)[:2]
        if self.ae_layer_norm is not None:
            x_w = self.ae_layer_norm(x_w)
        ha_w = self._attn_pool(x_w, self.pool_w).float()

        # ── Run unfrozen HuBERT layers ───────────────────────
        x_h = hubert_f.to(dev)
        for layer in self.he_unfrozen:
            x_h = layer(x_h, attention_mask=None)[0]
        if self.he_layer_norm is not None:
            x_h = self.he_layer_norm(x_h)
        ha_h = self._attn_pool(x_h, self.pool_h).float()

        # ── Text encoder ─────────────────────────────────────
        ht = self.te(
            input_ids.to(dev),
            attention_mask=attention_mask.to(dev),
        ).last_hidden_state[:, 0, :].float()

        # ── Cross-attention fusion ───────────────────────────
        ha_avg  = (ha_w + ha_h) / 2
        ha_ca   = F.relu(self.ap(ha_avg))
        ht_ca   = F.relu(self.tp(ht))
        attn, _ = self.ca(
            ht_ca.unsqueeze(1), ha_ca.unsqueeze(1), ha_ca.unsqueeze(1))
        ca_feat = torch.cat([ht_ca, ha_ca, attn.squeeze(1)], dim=1)

        # ── QFL: sum dual-audio → keeps MD = 2*proj_dim ─────
        qfl_audio = self.qfl_ap_w(ha_w) + self.qfl_ap_h(ha_h)
        qfl_text  = self.qfl_tp(ht)
        qfl_feat  = self.qfl(torch.cat([qfl_audio, qfl_text], dim=1))

        return self.clf(torch.cat([ca_feat, qfl_feat], dim=1))
