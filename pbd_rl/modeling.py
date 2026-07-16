from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from pxdesign.model.pxdesign import ProtenixDesign


@dataclass
class DiffusionLossOutput:
    per_sample_loss: torch.Tensor
    sigma: torch.Tensor


@dataclass
class ConditionEmbeddingCache:
    input_feature_dict: dict[str, Any]
    s_inputs: torch.Tensor
    s_trunk: torch.Tensor
    z_trunk: torch.Tensor


class PXDPOModel(ProtenixDesign):
    """
    PXDesign-based model for diffusion preference optimization.

    Key idea:
    - Use -L_diffusion as a surrogate score of log-probability.
    - Reuse PXDesign condition embedding + diffusion denoiser.
    """

    def _repeat_batch(self, obj: Any, repeats: int) -> Any:
        if torch.is_tensor(obj):
            return obj.repeat_interleave(repeats, dim=0)
        if isinstance(obj, dict):
            return {k: self._repeat_batch(v, repeats) for k, v in obj.items()}
        if isinstance(obj, list):
            repeated: list[Any] = []
            for v in obj:
                repeated.extend([v] * repeats)
            return repeated
        if isinstance(obj, tuple):
            repeated = []
            for v in obj:
                repeated.extend([v] * repeats)
            return tuple(repeated)
        return obj

    def encode_condition(
        self, input_feature_dict: dict[str, Any]
    ) -> ConditionEmbeddingCache:
        s_inputs, s, z = self.get_condition_embedding(input_feature_dict=input_feature_dict)
        return ConditionEmbeddingCache(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
        )

    def _sample_sigmas(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        num_noise_levels: int,
        num_parallel_steps: int = 1,
    ) -> torch.Tensor:
        noise_schedule = self.inference_noise_scheduler(
            N_step=num_noise_levels,
            device=device,
            dtype=dtype,
        )
        # Last element is forced to 0 by scheduler, so sample from non-zero steps.
        idx = torch.randint(
            low=0,
            high=noise_schedule.shape[0] - 1,
            size=(batch_size, num_parallel_steps),
            device=device,
        )
        return noise_schedule[idx]

    def _feature_dict_for_diffusion_samples(
        self, input_feature_dict: dict[str, Any], num_samples: int
    ) -> dict[str, Any]:
        if num_samples == 1:
            return input_feature_dict
        atom_to_token_idx = input_feature_dict.get("atom_to_token_idx")
        if (
            torch.is_tensor(atom_to_token_idx)
            and atom_to_token_idx.dim() == 2
            and atom_to_token_idx.shape[0] == 1
        ):
            out = dict(input_feature_dict)
            out["atom_to_token_idx"] = atom_to_token_idx[0]
            return out
        return input_feature_dict

    def diffusion_recon_loss_multi_candidates(
        self,
        input_feature_dict: dict[str, Any],
        x0: torch.Tensor,  # [B, C, N_atom, 3]
        atom_mask: torch.Tensor,  # [B, N_atom] or [N_atom]
        binder_mask: torch.Tensor | None = None,  # [B, N_atom] or [N_atom]
        candidate_valid_mask: torch.Tensor | None = None,  # [B, C]
        num_noise_levels: int = 400,
        num_parallel_steps: int = 1,
        use_binder_weight: bool = True,
        binder_weight: float = 1.0,
        non_binder_weight: float = 0.2,
        fixed_noise: torch.Tensor | None = None,  # [B, K, N_atom, 3]
        fixed_sigma: torch.Tensor | None = None,  # [B, K]
        condition_cache: ConditionEmbeddingCache | None = None,
    ) -> DiffusionLossOutput:
        if x0.dim() != 4:
            raise ValueError(f"x0 must be [B, C, N_atom, 3], got {tuple(x0.shape)}")

        B, C, N_atom, _ = x0.shape
        device = x0.device
        dtype = x0.dtype

        if condition_cache is None:
            condition_cache = self.encode_condition(input_feature_dict=input_feature_dict)

        sigma = fixed_sigma if fixed_sigma is not None else self._sample_sigmas(
            device=device,
            dtype=dtype,
            batch_size=B,
            num_noise_levels=num_noise_levels,
            num_parallel_steps=num_parallel_steps,
        )
        if fixed_noise is None:
            noise = torch.randn(
                B, num_parallel_steps, x0.shape[2], x0.shape[3], device=device, dtype=dtype
            )
        else:
            noise = fixed_noise
        if sigma.dim() != 2:
            raise ValueError(f"sigma must be [B, K], got {tuple(sigma.shape)}")
        if noise.dim() != 4:
            raise ValueError(f"noise must be [B, K, N_atom, 3], got {tuple(noise.shape)}")
        K = sigma.shape[1]
        if noise.shape[1] != K:
            raise ValueError(f"noise parallel dim {noise.shape[1]} != sigma parallel dim {K}")
        if noise.shape[2] != N_atom:
            raise ValueError(
                f"noise atom dim {noise.shape[2]} != candidate atom dim {N_atom}"
            )

        x0_k = x0[:, :, None, :, :].expand(-1, -1, K, -1, -1).reshape(
            B, C * K, N_atom, 3
        )
        sigma_flat = sigma[:, None, :].expand(-1, C, -1).reshape(B, C * K)
        noise_flat = noise[:, None, :, :, :].expand(-1, C, -1, -1, -1).reshape(
            B, C * K, N_atom, 3
        )
        x_t = x0_k + sigma_flat[:, :, None, None] * noise_flat
        sample_feature_dict = self._feature_dict_for_diffusion_samples(
            condition_cache.input_feature_dict,
            num_samples=C * K,
        )
        x_denoised = self.diffusion_module(
            x_noisy=x_t,
            t_hat_noise_level=sigma_flat,
            input_feature_dict=sample_feature_dict,
            s_inputs=condition_cache.s_inputs,
            s_trunk=condition_cache.s_trunk,
            z_trunk=condition_cache.z_trunk,
            chunk_size=None,
            inplace_safe=True,
        )

        # [B, C, K, N_atom]
        mse = F.mse_loss(x_denoised, x0_k, reduction="none").mean(dim=-1)
        mse = mse.reshape(B, C, K, N_atom)

        if atom_mask.dim() == 1:
            atom_mask = atom_mask.unsqueeze(0)
        atom_mask = atom_mask.to(device=device, dtype=mse.dtype)[:, None, None, :]
        atom_mask = atom_mask.expand(-1, C, K, -1)

        if binder_mask is not None and use_binder_weight:
            if binder_mask.dim() == 1:
                binder_mask = binder_mask.unsqueeze(0)
            binder_mask = binder_mask.to(device=device, dtype=mse.dtype)
            binder_mask = binder_mask[:, None, None, :].expand(-1, C, K, -1)
            atom_weights = (
                binder_weight * binder_mask + non_binder_weight * (1.0 - binder_mask)
            ) * atom_mask
        else:
            atom_weights = atom_mask

        # first aggregate atoms -> [B, C, K], then aggregate over parallel-noise K -> [B, C]
        denom = atom_weights.sum(dim=-1).clamp_min(1e-6)
        per_step = (mse * atom_weights).sum(dim=-1) / denom
        per_sample = per_step.mean(dim=-1)
        if candidate_valid_mask is not None:
            per_sample = per_sample * candidate_valid_mask.to(
                device=device, dtype=per_sample.dtype
            )
        return DiffusionLossOutput(per_sample_loss=per_sample, sigma=sigma)

    def diffusion_recon_loss(
        self,
        input_feature_dict: dict[str, Any],
        x0: torch.Tensor,  # [B, N_atom, 3]
        atom_mask: torch.Tensor,  # [B, N_atom] or [N_atom]
        binder_mask: torch.Tensor | None = None,  # [B, N_atom] or [N_atom]
        num_noise_levels: int = 400,
        num_parallel_steps: int = 1,
        use_binder_weight: bool = True,
        binder_weight: float = 1.0,
        non_binder_weight: float = 0.2,
        fixed_noise: torch.Tensor | None = None,  # [B, K, N_atom, 3]
        fixed_sigma: torch.Tensor | None = None,  # [B, K]
        condition_cache: ConditionEmbeddingCache | None = None,
    ) -> DiffusionLossOutput:
        if x0.dim() != 3:
            raise ValueError(f"x0 must be [B, N_atom, 3], got {tuple(x0.shape)}")
        out = self.diffusion_recon_loss_multi_candidates(
            input_feature_dict=input_feature_dict,
            x0=x0.unsqueeze(1),
            atom_mask=atom_mask,
            binder_mask=binder_mask,
            candidate_valid_mask=torch.ones(
                x0.shape[0], 1, device=x0.device, dtype=x0.dtype
            ),
            num_noise_levels=num_noise_levels,
            num_parallel_steps=num_parallel_steps,
            use_binder_weight=use_binder_weight,
            binder_weight=binder_weight,
            non_binder_weight=non_binder_weight,
            fixed_noise=fixed_noise,
            fixed_sigma=fixed_sigma,
            condition_cache=condition_cache,
        )
        return DiffusionLossOutput(
            per_sample_loss=out.per_sample_loss[:, 0],
            sigma=out.sigma,
        )
