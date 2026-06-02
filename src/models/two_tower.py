"""Two-tower model for click-through-rate prediction.

Architecture: UserTower and MovieTower each encode their inputs into a
D-dimensional embedding. Score = sigmoid(dot_product(user_emb, movie_emb)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """Build the feed-forward MLP used inside each tower.

    The network maps: in_dim → hidden_dims[0] → ... → hidden_dims[-1] → out_dim.

    TODO(human): implement this function.
    Consider: activation function (ReLU, GELU, SiLU?), regularization (Dropout
    and at what rate?), and normalization (BatchNorm1d, LayerNorm, or none?).
    Should the last layer have an activation? Should dropout come before or after
    the activation? There is no single right answer — your choices will affect
    training stability and generalization.
    """
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for dim in hidden_dims:
        layers += [nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.1)]
        prev_dim = dim
    layers += [nn.Linear(prev_dim, out_dim), nn.Tanh()]
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    """Encodes (user_id, behavior_features) → D-dimensional embedding."""

    def __init__(
        self,
        n_users: int,
        embed_dim: int,
        behavior_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.mlp = _build_mlp(embed_dim + behavior_dim, hidden_dims, output_dim)

    def forward(self, user_ids: Tensor, behavior: Tensor) -> Tensor:
        """Return user embedding [B, output_dim]."""
        x = torch.cat([self.embed(user_ids), behavior], dim=-1)
        return self.mlp(x)


class MovieTower(nn.Module):
    """Encodes (movie_id, meta_features) → D-dimensional embedding."""

    def __init__(
        self,
        n_movies: int,
        embed_dim: int,
        meta_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_movies + 1, embed_dim, padding_idx=0)
        self.mlp = _build_mlp(embed_dim + meta_dim, hidden_dims, output_dim)

    def forward(self, movie_ids: Tensor, meta: Tensor) -> Tensor:
        """Return movie embedding [B, output_dim]."""
        x = torch.cat([self.embed(movie_ids), meta], dim=-1)
        return self.mlp(x)


class TwoTowerModel(nn.Module):
    """Combines UserTower and MovieTower; predicts P(click) via dot product."""

    def __init__(self, user_tower: UserTower, movie_tower: MovieTower) -> None:
        super().__init__()
        self.user_tower = user_tower
        self.movie_tower = movie_tower

    def encode_user(self, user_ids: Tensor, behavior: Tensor) -> Tensor:
        """Return user embedding [B, D]. Used in serving to pre-compute user vector."""
        return self.user_tower(user_ids, behavior)

    def encode_movie(self, movie_ids: Tensor, meta: Tensor) -> Tensor:
        """Return movie embedding [B, D]. Used in serving to pre-compute movie vectors."""
        return self.movie_tower(movie_ids, meta)

    def forward(
        self, user_ids: Tensor, behavior: Tensor, movie_ids: Tensor, meta: Tensor
    ) -> Tensor:
        """Return P(click) scores [B], values in (0, 1)."""
        u = self.encode_user(user_ids, behavior)
        m = self.encode_movie(movie_ids, meta)
        return torch.sigmoid((u * m).sum(dim=-1))
