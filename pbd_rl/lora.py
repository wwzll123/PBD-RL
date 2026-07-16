from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.base = base
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return out + lora * self.scaling

    def merged_weight(self) -> torch.Tensor:
        delta = torch.matmul(self.lora_b, self.lora_a) * self.scaling
        return self.base.weight.detach() + delta.detach().to(
            device=self.base.weight.device, dtype=self.base.weight.dtype
        )


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_prefixes: tuple[str, ...] = ("diffusion_module",)
    target_suffixes: tuple[str, ...] = ("",)


def _matches(name: str, prefixes: tuple[str, ...], suffixes: tuple[str, ...]) -> bool:
    if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
        return False
    return (not suffixes) or any(name.endswith(suffix) for suffix in suffixes)


def inject_lora_linear(model: nn.Module, cfg: LoRAConfig) -> int:
    replaced = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _matches(
                full_name, cfg.target_prefixes, cfg.target_suffixes
            ):
                setattr(
                    module,
                    child_name,
                    LoRALinear(
                        child,
                        rank=cfg.rank,
                        alpha=cfg.alpha,
                        dropout=cfg.dropout,
                    ),
                )
                replaced += 1
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = ".lora_a" in name or ".lora_b" in name


def merged_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"{name}."
        out[f"{name}.weight"] = module.merged_weight().cpu()
        if module.base.bias is not None:
            out[f"{name}.bias"] = module.base.bias.detach().cpu()
        consumed.update(
            {
                prefix + "base.weight",
                prefix + "base.bias",
                prefix + "lora_a",
                prefix + "lora_b",
            }
        )

    for key, value in state.items():
        if key in consumed:
            continue
        if ".base." in key or key.endswith(".lora_a") or key.endswith(".lora_b"):
            continue
        out[key] = value.detach().cpu()
    return out
