from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


class ConstraintProjector:
    def __init__(
        self,
        n_nodes: int,
        *,
        home_size: int | None = None,
        positive_target: float = 1.0,
        negative_scale: float = 1.25,
        max_iters: int = 1,
        tol: float = 1e-3,
    ) -> None:
        self.n_nodes = n_nodes
        self.home_size = home_size if home_size is not None else n_nodes // 2
        self.positive_target = positive_target
        self.negative_scale = negative_scale
        self.max_iters = max_iters
        self.tol = tol

        nh = self.home_size
        self.top_left = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
        self.top_left[:nh, :nh] = 1
        self.top_left.fill_diagonal_(0)

        self.bottom_right = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
        self.bottom_right[nh:, nh:] = 1
        self.bottom_right.fill_diagonal_(0)

        self.top_right = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
        self.top_right[:nh, nh:] = 1

        self.bottom_left = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
        self.bottom_left[nh:, :nh] = 1

        self.positive_mask = (self.top_left | self.bottom_right).float()
        self.negative_mask = (self.top_right | self.bottom_left).float()

    def _project_single(self, matrix: torch.Tensor) -> torch.Tensor:
        projected = matrix.clone().float().cpu()
        nh = self.home_size
        away_size = self.n_nodes - nh
        for _ in range(self.max_iters):
            previous = projected.clone()
            projected.fill_diagonal_(0)
            projected[:nh, :nh] *= self.positive_target / (projected[:nh, :nh].sum() + 1e-8)
            projected[nh:, nh:] *= self.positive_target / (projected[nh:, nh:].sum() + 1e-8)

            row_pos = (projected * self.positive_mask).sum(dim=1, keepdim=True)
            row_neg = (projected * self.negative_mask).sum(dim=1, keepdim=True)
            scale = -self.negative_scale * row_pos / (row_neg + 1e-8)
            projected = projected * (self.negative_mask * scale + (1 - self.negative_mask))

            col_pos = (projected * self.positive_mask).sum(dim=0, keepdim=True)
            col_neg = (projected * self.negative_mask).sum(dim=0, keepdim=True)
            scale = -self.negative_scale * col_pos / (col_neg + 1e-8)
            projected = projected * (self.negative_mask * scale + (1 - self.negative_mask))
            projected = 0.5 * (projected + projected.transpose(0, 1))
            if torch.norm(projected - previous) < self.tol:
                break

        projected[:nh, :nh] /= projected[:nh, :nh].sum() * (1 / max(nh**2, 1))
        projected[nh:, nh:] /= projected[nh:, nh:].sum() * (1 / max(away_size**2, 1))
        projected[:nh, nh:] /= -projected[:nh, nh:].sum() * (1 / max(nh * away_size, 1))
        projected[nh:, :nh] /= -projected[nh:, :nh].sum() * (1 / max(nh * away_size, 1))
        return projected

    def project(self, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 2:
            return self._project_single(adjacency)
        return torch.stack([self._project_single(matrix) for matrix in adjacency], dim=0)


class _StackelbergState(nn.Module):
    def __init__(
        self,
        initial_state: torch.Tensor,
        *,
        home_size: int,
        leader_mask: torch.Tensor,
        follower_mask: torch.Tensor,
        projector: ConstraintProjector,
    ) -> None:
        super().__init__()
        self.state = nn.Parameter(initial_state)
        self.home_size = home_size
        self.leader_mask = leader_mask
        self.follower_mask = follower_mask
        self.projector = projector
        self.relu = nn.ReLU()

    def build(self) -> torch.Tensor:
        matrix = self.state.clone()
        nh = self.home_size
        matrix[:, :nh, :nh] = self.relu(matrix[:, :nh, :nh])
        matrix[:, nh:, nh:] = self.relu(matrix[:, nh:, nh:])
        matrix[:, :nh, nh:] = -self.relu(-matrix[:, :nh, nh:])
        matrix[:, nh:, :nh] = -self.relu(-matrix[:, nh:, :nh])
        return self.projector.project(matrix)

    def leader_view(self, frozen_mask: torch.Tensor) -> torch.Tensor:
        matrix = self.state.clone()
        leader_trainable = self.leader_mask - frozen_mask
        return matrix * leader_trainable + matrix.detach() * (1 - leader_trainable)

    def follower_view(self) -> torch.Tensor:
        matrix = self.build()
        return matrix * self.follower_mask + matrix.detach() * (1 - self.follower_mask)


@dataclass
class StackelbergResult:
    score: torch.Tensor
    adjacency: torch.Tensor
    leader_optimizer: torch.optim.Optimizer
    follower_optimizer: torch.optim.Optimizer


def stackelberg_optimize(
    features: torch.Tensor,
    models: Sequence[nn.Module],
    frozen_mask: torch.Tensor,
    *,
    initial_adjacency: torch.Tensor | None = None,
    home_size: int | None = None,
    steps: int | None = None,
    leader_lr: float = 1e-5,
    follower_lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> StackelbergResult:
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, nodes, features]")

    device = device or features.device
    batch_size, n_nodes, _ = features.shape
    home_size = home_size if home_size is not None else n_nodes // 2
    if n_nodes % 2 != 0:
        raise ValueError("stackelberg_optimize expects an even number of nodes")

    projector = ConstraintProjector(n_nodes, home_size=home_size)
    leader_mask = torch.zeros(n_nodes, n_nodes, device=device, dtype=torch.int32).unsqueeze(0)
    leader_mask[:, :home_size, :home_size] = True
    follower_mask = 1 - leader_mask

    if initial_adjacency is None:
        initial_state = torch.ones(batch_size, n_nodes, n_nodes, device=device)
        initial_state /= initial_state.sum()
        initial_state[:, home_size:, :home_size] *= -1.25
        initial_state[:, :home_size, home_size:] *= -1.25
    else:
        initial_state = initial_adjacency.clone().to(device)

    initial_state[frozen_mask == 1] = 0
    state = _StackelbergState(
        initial_state,
        home_size=home_size,
        leader_mask=leader_mask,
        follower_mask=follower_mask,
        projector=projector,
    ).to(device)
    leader_optimizer = torch.optim.Adamax(
        [state.state],
        lr=leader_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.99),
        eps=1e-5,
    )
    follower_optimizer = torch.optim.Adamax(
        [state.state],
        lr=follower_lr,
        weight_decay=weight_decay,
        betas=(0.8, 0.98),
        eps=1e-5,
    )

    total_steps = steps if steps is not None else (50 if initial_adjacency is not None else 100)
    leader_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(leader_optimizer, T_max=total_steps)
    follower_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        follower_optimizer, T_max=max(total_steps // 5, 1)
    )

    features = features.to(device)
    frozen_mask = frozen_mask.to(device)
    home_mask = torch.zeros(n_nodes, device=device, dtype=torch.bool)
    home_mask[:home_size] = True
    home_mask = home_mask.unsqueeze(0).expand(batch_size, -1)

    for step in range(total_steps):
        leader_view = state.leader_view(frozen_mask)
        leader_loss = torch.stack([-model(leader_view, features, home_mask).mean() for model in models]).mean()
        leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_([state.state], max_norm=1.0)
        leader_optimizer.step()

        if (step + 1) % 5 == 0:
            follower_view = state.follower_view()
            follower_loss = torch.stack([model(follower_view, features, home_mask).mean() for model in models]).mean()
            follower_optimizer.zero_grad()
            follower_loss.backward()
            torch.nn.utils.clip_grad_norm_([state.state], max_norm=1.0)
            follower_optimizer.step()
            follower_scheduler.step()

        leader_scheduler.step()

    adjacency = state.build().detach()
    score = torch.stack([model(adjacency, features, home_mask) for model in models]).mean(dim=0).cpu()
    return StackelbergResult(
        score=score,
        adjacency=adjacency.cpu(),
        leader_optimizer=leader_optimizer,
        follower_optimizer=follower_optimizer,
    )
