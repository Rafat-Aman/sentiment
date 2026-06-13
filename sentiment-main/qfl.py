# ============================================================
# qfl.py — Quantum Fusion Layer (unchanged from original)
# ============================================================
import math
import torch
import torch.nn as nn


class QuantumFusionLayer(nn.Module):
    def __init__(self, MD: int, P: int = 1, n_pqc_layers: int = 5):
        super().__init__()
        self.MD           = MD
        self.P            = P
        self.n_pqc_layers = n_pqc_layers
        self.n_index      = max(1, math.ceil(math.log2(MD + 1)))
        self.n_qubits     = self.n_index + 1
        self.state_dim    = 2 ** self.n_qubits
        self.max_feat     = min(MD, 2 ** self.n_index - 1)
        self.out_dim      = 3 * self.n_qubits
        self.pqc_weights  = nn.Parameter(
            torch.empty(P + 1, n_pqc_layers, self.n_index, 3).uniform_(-0.1, 0.1))
        self._cache_dev   = None
        self._cnot_mats   = None
        self._obs_mats    = None

    def _rebuild_cache(self, dev):
        d, nq, ni = self.state_dim, self.n_qubits, self.n_index

        def _make_cnot(ctrl, tgt):
            perm = torch.arange(d)
            cb, tb = nq - 1 - ctrl, nq - 1 - tgt
            mask = (perm >> cb) & 1
            perm = perm ^ (mask << tb)
            mat = torch.zeros(d, d, dtype=torch.cfloat, device=dev)
            mat[perm, torch.arange(d)] = 1.0
            return mat

        cnots = [_make_cnot(k, k + 1) for k in range(ni - 1)]
        if ni > 2:
            cnots.append(_make_cnot(ni - 1, 0))
        self._cnot_mats = cnots

        Z2 = torch.tensor([[1., 0.], [0., -1.]],  dtype=torch.cfloat, device=dev)
        X2 = torch.tensor([[0., 1.], [1.,  0.]],  dtype=torch.cfloat, device=dev)
        Y2 = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.cfloat, device=dev)

        def _embed(g2, qubit):
            I = torch.eye(2, dtype=g2.dtype, device=dev)
            out = g2 if qubit == 0 else I.clone()
            for q in range(1, nq):
                out = torch.kron(out, g2 if q == qubit else I)
            return out

        obs = [_embed(p, q) for q in range(nq) for p in (Z2, X2, Y2)]
        self._obs_mats  = torch.stack(obs)
        self._cache_dev = str(dev)

    def _ensure_cache(self, dev):
        if self._cache_dev != str(dev) or self._cnot_mats is None:
            self._rebuild_cache(dev)

    @staticmethod
    def _rx(a):
        c, s = torch.cos(a / 2), torch.sin(a / 2); z = torch.zeros_like(c)
        return torch.complex(
            torch.stack([torch.stack([c, z]), torch.stack([z, c])]),
            torch.stack([torch.stack([z, -s]), torch.stack([-s, z])]))

    @staticmethod
    def _ry(a):
        c, s = torch.cos(a / 2), torch.sin(a / 2)
        return torch.complex(
            torch.stack([torch.stack([c, -s]), torch.stack([s, c])]),
            torch.zeros(2, 2, device=a.device))

    @staticmethod
    def _rz(a):
        c, s = torch.cos(a / 2), torch.sin(a / 2); z = torch.zeros_like(c)
        return torch.complex(
            torch.stack([torch.stack([c, z]), torch.stack([z, c])]),
            torch.stack([torch.stack([-s, z]), torch.stack([z, s])]))

    def _embed_gate(self, g2, qubit, dev):
        nq = self.n_qubits
        I = torch.eye(2, dtype=g2.dtype, device=dev)
        out = g2 if qubit == 0 else I.clone()
        for q in range(1, nq):
            out = torch.kron(out, g2 if q == qubit else I)
        return out

    def _ansatz(self, block_idx, dev):
        U = torch.eye(self.state_dim, dtype=torch.cfloat, device=dev)
        w = self.pqc_weights[block_idx]
        for layer in range(self.n_pqc_layers):
            for q in range(self.n_index):
                Rx = self._embed_gate(self._rx(w[layer, q, 0]), q, dev)
                Ry = self._embed_gate(self._ry(w[layer, q, 1]), q, dev)
                Rz = self._embed_gate(self._rz(w[layer, q, 2]), q, dev)
                U = Rz @ Ry @ Rx @ U
            for cnot in self._cnot_mats:
                U = cnot @ U
        return U

    def _state_prep_phases(self, x):
        B, dev = x.shape[0], x.device
        diag = torch.ones(B, self.state_dim, dtype=torch.cfloat, device=dev)
        for j in range(self.max_feat):
            phi = torch.arccos(x[:, j].clamp(-1 + 1e-6, 1 - 1e-6))
            diag[:, j + 1] = torch.complex(torch.cos(phi), torch.sin(phi))
        return diag

    def forward(self, x):
        B, dev = x.shape[0], x.device
        self._ensure_cache(dev)
        x_norm = torch.tanh(x)
        Us = [self._ansatz(p, dev) for p in range(self.P + 1)]
        psi = torch.zeros(B, self.state_dim, dtype=torch.cfloat, device=dev)
        psi[:, 0] = 1.0
        psi = torch.einsum('ij,bj->bi', Us[self.P], psi)
        for p in range(self.P - 1, -1, -1):
            psi = psi * self._state_prep_phases(x_norm)
            psi = torch.einsum('ij,bj->bi', Us[p], psi)
        obs_psi = torch.einsum('kij,bj->bki', self._obs_mats, psi)
        exp_vals = (psi.conj().unsqueeze(1) * obs_psi).sum(-1).real
        return exp_vals
