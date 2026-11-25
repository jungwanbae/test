"""CrossAssetTemporalNet — 자산별 시계열 인코더(GRU) + 자산 간 셀프어텐션(Transformer).

  입력 x: [배치, 자산, 시퀀스, 피처]
  1) 자산마다 같은 가중치의 GRU 로 최근 L 바의 피처 시퀀스를 요약(파라미터 공유 → 과적합 억제)
  2) 자산 임베딩을 더한 뒤 Transformer 인코더로 자산 간 관계(위험선호/회피 국면)를 학습
  3) 자산별 점수 출력 → 다음 기간 '변동성 조정 초과수익'의 상대 순위를 예측

손실 = Huber(회귀) + λ·ListNet(순위). 결측 자산은 마스크로 제외.
"""
from __future__ import annotations

import torch
from torch import nn


class CrossAssetTemporalNet(nn.Module):
    def __init__(self, n_features: int, n_assets: int, hidden: int = 32, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.asset_emb = nn.Embedding(n_assets, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
                                           dropout=dropout, batch_first=True)
        self.cross = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, A, L, F = x.shape
        h = self.inp(x.reshape(B * A, L, F))
        _, hn = self.gru(h)
        h = hn[-1].reshape(B, A, -1) + self.asset_emb.weight[:A].unsqueeze(0)
        if mask is not None and mask.any():
            # 모든 자산이 마스크되면 어텐션이 NaN → 그런 행은 마스크 해제(출력은 어차피 버림)
            all_masked = mask.all(dim=1, keepdim=True)
            mask = mask & ~all_masked
        h = self.cross(h, src_key_padding_mask=mask)
        return self.head(h).squeeze(-1)


def ranking_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """ListNet: softmax(target) 과 log_softmax(pred) 의 교차엔트로피(유효 자산만)."""
    neg = torch.finfo(pred.dtype).min / 4
    p = pred.masked_fill(~valid, neg)
    t = (target / temp).masked_fill(~valid, neg)
    loss = -(torch.softmax(t, dim=1) * torch.log_softmax(p, dim=1)).masked_fill(~valid, 0.0).sum(dim=1)
    rows = valid.sum(dim=1) >= 2
    return loss[rows].mean() if rows.any() else pred.sum() * 0.0


def combined_loss(pred, target, valid, rank_weight: float = 0.5, delta: float = 1.0):
    hub = nn.functional.huber_loss(pred[valid], target[valid], delta=delta) if valid.any() else pred.sum() * 0.0
    return hub + rank_weight * ranking_loss(pred, target, valid)
