"""Live region-integrator co-diagnosis (#2 dx) for interf1.

region encoder (ABMIL) + integrator trained end to end for dx-set prediction
Splits a WSI's tiles into 5mm grid regions, the joint encoder embeds each region, 
and the integrator (queried by a learned per-base-dx #1 anchor, selected by the single-label #1-diagnosis model)
predicts the reported dx SET.
Agreement-gated: emits the residual dxs above threshold when the integrator's top == the #1 diagnosis, mapped to A_VOCAB indices.

Handles GENERAL / novel co-dx (generalization). DCIS/CIS are ALSO covered by their dedicated
cofinding tile-heads (better on those specific entities) - the integrator + heads both feed
the dx chain (deduped) and the in-situ subtree. Never raises -> [] on any failure.
"""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reg2026.aggregate.abmil import ABMIL, ABMILConfig


def _base(d: str) -> str:
    d = re.sub(r",?\s*grade\s+\S+\s*$", "", d, flags=re.I)
    d = re.sub(r",?\s*(well|moderately|poorly|undifferentiated)\s+differentiated\s*$", "", d, flags=re.I)
    return re.sub(r",?\s*(high|low|intermediate)[- ]grade\s*$", "", d, flags=re.I).strip().lower()


class _AbmilClf(nn.Module):
    """Single-label #1-diagnosis model (ABMIL 1536->1024 + classifier head)."""

    def __init__(self, n: int):
        super().__init__()
        self.abmil = ABMIL(ABMILConfig())
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.25), nn.Linear(256, n)
        )

    def forward(self, x, msk):
        e, _ = self.abmil(x, msk)
        return self.classifier(e)


class _RegionCoDxModel(nn.Module):
    """Co-dx model, trained end to end ('joint'). Two-level MIL: each 5mm grid region is a BAG of its
    tiles, ABMIL-pooled (enc) into ONE region embedding; then the #1-diagnosis anchor prototype
    (proto[a1id]) cross-attends (xa) over the region embeddings to predict the co-diagnosis SET
    (multilabel over gvocab, via head). proto = per-base-dx anchor prototypes."""

    def __init__(self, B: int, G: int, n: int, d: int = 256, h: int = 4):
        super().__init__()
        self.enc = ABMIL(ABMILConfig())
        self.proto = nn.Embedding(B, 1024)
        self.rp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d), nn.GELU())
        self.qp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d), nn.GELU())
        self.xa = nn.MultiheadAttention(d, h, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, G))
        self.rcls = nn.Linear(1024, n)  # region-cls aux (loaded but unused at inference)

    @torch.no_grad()
    def encode(self, bags, dev):
        embs = []
        for f in bags:
            x = f.to(dev).unsqueeze(0)
            e, _ = self.enc(x, torch.ones(1, x.shape[1], dtype=torch.bool, device=dev))
            embs.append(e[0])
        return torch.stack(embs)

    @torch.no_grad()
    def forward(self, bags, a1id, dev):
        reg = self.encode(bags, dev).unsqueeze(0)
        proto = self.proto(torch.tensor([a1id], device=dev))
        K = self.rp(reg)
        q = self.qp(proto).unsqueeze(1)
        att, _ = self.xa(q, K, K)
        return self.head(torch.cat([self.qp(proto), att.squeeze(1)], -1))  # [1,G]


class CodxIntegrator:
    """Joint region-encoder + integrator; agreement-gated co-dx A_VOCAB indices per WSI."""

    REGION_PX = 10080   # region edge, level-0 px (@0.5 mpp = ~5mm)
    MIN_T = 40     # min tiles for a region to be kept (drop sparse regions)
    MAX_T = 1024   # max tiles per region (subsample the region's ABMIL bag)
    RMAX = 32      # max regions per slide (subsample if more; caps cross-attention input)
    TAU = 0.7      # sigmoid prob threshold to emit a co-dx

    def __init__(self, primary_ckpt, integrator_ckpt, a_vocab, device: str):
        self.dev = device
        primary_ck = torch.load(primary_ckpt, map_location="cpu", weights_only=False)
        self.dxv = primary_ck["dx_vocab"]
        self.primary = _AbmilClf(len(self.dxv))
        self.primary.load_state_dict(primary_ck["model_state"], strict=True)
        self.primary.eval().to(device)
        integ_ck = torch.load(integrator_ckpt, map_location="cpu", weights_only=False)
        self.bvocab, self.gvocab = integ_ck["bvocab"], integ_ck["gvocab"]
        self.b2i = {b: i for i, b in enumerate(self.bvocab)}
        self.codx_model = _RegionCoDxModel(len(self.bvocab), len(self.gvocab), len(self.dxv))
        self.codx_model.load_state_dict(integ_ck["state"], strict=True)
        self.codx_model.eval().to(device)
        # base-dx -> A_VOCAB idx (graded-exact preferred, else base match)
        self.b2a = {}
        for i, a in enumerate(a_vocab):
            ba = _base(a)
            if ba not in self.b2a or a.strip().lower() == ba:
                self.b2a[ba] = i

    @torch.no_grad()
    def _dx1_base(self, feats: np.ndarray) -> str:
        f = feats
        if len(f) > 2048:
            f = f[np.random.default_rng(0).choice(len(f), 2048, replace=False)]
        x = torch.from_numpy(f).float().to(self.dev).unsqueeze(0)
        sm = torch.softmax(self.primary(x, torch.ones(1, x.shape[1], dtype=torch.bool, device=self.dev))[0], -1)
        return _base(self.dxv[int(sm.argmax())])

    @torch.no_grad()
    def compute(self, coords: np.ndarray, feats: np.ndarray) -> list[int]:
        """Agreement-gated co-dx A_VOCAB indices for this WSI ([] on any failure)."""
        try:
            cx = ((coords[:, 0] - coords[:, 0].min()) // self.REGION_PX).astype(int)
            cy = ((coords[:, 1] - coords[:, 1].min()) // self.REGION_PX).astype(int)
            region_id = cx * 10000 + cy
            ri = [np.sort(np.where(region_id == u)[0]) for u in np.unique(region_id)]
            ri = [idx for idx in ri if len(idx) >= self.MIN_T]
            if not ri:
                return []
            if len(ri) > self.RMAX:
                sel = np.random.default_rng(0).choice(len(ri), self.RMAX, replace=False)
                ri = [ri[j] for j in sel]
            bags = []
            for t in ri:
                f = feats[t]
                if len(f) > self.MAX_T:
                    f = f[np.random.default_rng(0).choice(len(f), self.MAX_T, replace=False)]
                bags.append(torch.from_numpy(f).float())
            dx1 = self._dx1_base(feats)
            a1id = self.b2i.get(dx1, 0)
            prob = torch.sigmoid(self.codx_model(bags, a1id, self.dev)[0]).cpu().numpy()
            if self.gvocab[int(prob.argmax())] != dx1:  # agreement gate
                return []
            return sorted(
                {self.b2a[self.gvocab[i]] for i in range(len(self.gvocab))
                 if prob[i] >= self.TAU and self.gvocab[i] != dx1 and self.gvocab[i] in self.b2a
                 and re.search(r"[A-Za-z]", self.gvocab[i])}  # skip degenerate no-letter entries (e.g. '3')
            )
        except Exception:  # noqa: BLE001
            import sys
            import traceback

            print("[codx_integrator] region integrator failed - returning [] (no co-dx):", file=sys.stderr)
            traceback.print_exc()
            return []


def load_codx_integrator(ckpts_dir, device: str) -> CodxIntegrator:
    from pathlib import Path

    from reg2026.data.embedding_pools import EmbeddingPools

    d = Path(ckpts_dir)
    return CodxIntegrator(
        primary_ckpt=d / "abmil_primary_dx.ckpt",
        integrator_ckpt=d / "cofinding" / "codx_region_integrator.ckpt",
        a_vocab=EmbeddingPools().A_VOCAB,
        device=device,
    )
