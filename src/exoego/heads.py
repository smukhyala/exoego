"""Trainable temporal head over frozen per-frame features.

A temporal model is not optional here. The two largest verb classes are
"pick up" (9,009 segments) and "put down" (8,184), which are the same frames in
opposite temporal order. Mean-pooling frame features cannot separate them, so
the head is a small transformer over the frame sequence with a learned position
embedding and a CLS token.
"""

import torch
import torch.nn as nn


class TemporalHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 384,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 24,
        proj_dim: int = 128,
        num_frames: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, num_classes)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        """(B, T, in_dim) -> (B, d_model) clip embedding."""
        batch_size = features.shape[0]
        tokens = self.input_proj(features)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])

    def forward(self, features: torch.Tensor):
        embedding = self.embed(features)
        logits = self.classifier(embedding)
        projected = self.projection(embedding)
        return embedding, logits, projected
