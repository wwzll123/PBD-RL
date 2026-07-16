import torch
import torch.nn.functional as F


def _reduce_loss(
    loss_vec: torch.Tensor, reduction: str
) -> torch.Tensor:
    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    return loss_vec


def diffusion_dpo_loss(
    policy_good: torch.Tensor,
    policy_bad: torch.Tensor,
    ref_good: torch.Tensor | None,
    ref_bad: torch.Tensor | None,
    beta: float = 1.0,
    reference_free: bool = False,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    policy_good/bad, ref_good/bad are diffusion losses L (lower is better), shape [B].
    score is defined as -L.
    """
    # Delta on diffusion losses. Since score = -L, this corresponds to score_diff.
    policy_delta = policy_bad - policy_good

    if reference_free or ref_good is None or ref_bad is None:
        logits = beta * policy_delta
    else:
        ref_delta = ref_bad - ref_good
        logits = beta * (policy_delta - ref_delta)

    loss_vec = -F.logsigmoid(logits)
    loss = _reduce_loss(loss_vec, reduction)

    metrics = {
        "policy_delta_mean": policy_delta.detach().mean(),
        "logits_mean": logits.detach().mean(),
        "dpo_loss_mean": loss_vec.detach().mean(),
    }
    return loss, metrics


def diffusion_dpo_loss_multi_candidate(
    policy_loss: torch.Tensor,
    ref_loss: torch.Tensor | None,
    candidate_rank: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None = None,
    beta: float = 1.0,
    reference_free: bool = False,
    loss_type: str = "top1_vs_rest",
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    policy_loss/ref_loss: [B, C], lower is better.
    candidate_rank: [B, C], smaller rank means more preferred.
    candidate_valid_mask: [B, C], 1 for valid candidates, 0 for padding.
    """
    if policy_loss.dim() != 2:
        raise ValueError(f"policy_loss must be [B, C], got {tuple(policy_loss.shape)}")
    if candidate_rank.shape != policy_loss.shape:
        raise ValueError(
            f"candidate_rank shape {tuple(candidate_rank.shape)} must match "
            f"policy_loss shape {tuple(policy_loss.shape)}"
        )
    if ref_loss is not None and ref_loss.shape != policy_loss.shape:
        raise ValueError(
            f"ref_loss shape {tuple(ref_loss.shape)} must match "
            f"policy_loss shape {tuple(policy_loss.shape)}"
        )

    valid = (
        torch.ones_like(policy_loss, dtype=torch.bool)
        if candidate_valid_mask is None
        else candidate_valid_mask.to(dtype=torch.bool)
    )
    rank_i = candidate_rank.unsqueeze(2)
    rank_j = candidate_rank.unsqueeze(1)
    valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1) & (rank_i < rank_j)

    if loss_type == "top1_vs_rest":
        best_rank = torch.where(valid, candidate_rank, torch.full_like(candidate_rank, 10**9))
        best_rank = best_rank.min(dim=1, keepdim=True).values
        valid_pairs = valid_pairs & (rank_i == best_rank.unsqueeze(2))
    elif loss_type == "all_pairs":
        pass
    else:
        raise ValueError(f"Unsupported multi-candidate loss_type: {loss_type}")

    if not valid_pairs.any():
        raise ValueError("No valid preference pairs found in multi-candidate batch.")

    policy_good = policy_loss.unsqueeze(2)
    policy_bad = policy_loss.unsqueeze(1)
    policy_delta = policy_bad - policy_good

    if reference_free or ref_loss is None:
        logits = beta * policy_delta
    else:
        ref_good = ref_loss.unsqueeze(2)
        ref_bad = ref_loss.unsqueeze(1)
        ref_delta = ref_bad - ref_good
        logits = beta * (policy_delta - ref_delta)

    loss_mat = -F.logsigmoid(logits)
    loss_vec = loss_mat[valid_pairs]
    logits_vec = logits[valid_pairs]
    policy_delta_vec = policy_delta[valid_pairs]
    loss = _reduce_loss(loss_vec, reduction)

    metrics = {
        "policy_delta_mean": policy_delta_vec.detach().mean(),
        "logits_mean": logits_vec.detach().mean(),
        "dpo_loss_mean": loss_vec.detach().mean(),
        "num_preference_pairs": valid_pairs.sum().detach().to(dtype=policy_loss.dtype),
    }
    return loss, metrics


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def diffusion_grpo_loss(
    policy_loss: torch.Tensor,
    ref_loss: torch.Tensor | None,
    candidate_score: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None = None,
    kl_coef: float = 0.02,
    advantage_eps: float = 1e-6,
    advantage_clip: float | None = 5.0,
    reward_clip: float | None = None,
    reference_free: bool = False,
    clip_range: float | None = 0.2,
    log_ratio_clip: float | None = 5.0,
    negative_advantage_weight: float = 1.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Offline group-relative policy objective for diffusion candidates.

    policy_loss/ref_loss are diffusion reconstruction losses L, shape [B, C].
    We use logp_surrogate = -L. Rewards are normalized within each target group
    to form GRPO-style advantages.
    """
    if policy_loss.dim() != 2:
        raise ValueError(f"policy_loss must be [B, C], got {tuple(policy_loss.shape)}")
    if candidate_score.shape != policy_loss.shape:
        raise ValueError(
            f"candidate_score shape {tuple(candidate_score.shape)} must match "
            f"policy_loss shape {tuple(policy_loss.shape)}"
        )
    if ref_loss is not None and ref_loss.shape != policy_loss.shape:
        raise ValueError(
            f"ref_loss shape {tuple(ref_loss.shape)} must match "
            f"policy_loss shape {tuple(policy_loss.shape)}"
        )

    valid = (
        torch.ones_like(policy_loss, dtype=torch.bool)
        if candidate_valid_mask is None
        else candidate_valid_mask.to(dtype=torch.bool)
    )
    valid = valid & torch.isfinite(candidate_score) & torch.isfinite(policy_loss)
    if not valid.any():
        raise ValueError("No valid candidates found for GRPO loss.")

    reward = candidate_score.to(dtype=policy_loss.dtype)
    if reward_clip is not None:
        reward = reward.clamp(min=-float(reward_clip), max=float(reward_clip))

    reward_for_stats = torch.where(valid, reward, torch.zeros_like(reward))
    counts = valid.sum(dim=1, keepdim=True).to(dtype=policy_loss.dtype).clamp_min(1.0)
    reward_mean = reward_for_stats.sum(dim=1, keepdim=True) / counts
    centered = torch.where(valid, reward - reward_mean, torch.zeros_like(reward))
    reward_var = (centered.square().sum(dim=1, keepdim=True) / counts).clamp_min(0.0)
    reward_std = reward_var.sqrt()
    advantage = centered / reward_std.clamp_min(float(advantage_eps))
    if advantage_clip is not None:
        advantage = advantage.clamp(min=-float(advantage_clip), max=float(advantage_clip))
    advantage = torch.where(valid, advantage, torch.zeros_like(advantage)).detach()

    if reference_free or ref_loss is None:
        raise ValueError(
            "GRPO training must use a frozen reference model. "
            "Set dpo.reference_free=false to avoid unbounded diffusion drift."
        )
    else:
        policy_logp = -policy_loss
        ref_logp = -ref_loss
        logp_diff = policy_logp - ref_logp
        kl_loss = _masked_mean(logp_diff.square(), valid)

    clipped_log_ratio = logp_diff
    if log_ratio_clip is not None:
        clipped_log_ratio = clipped_log_ratio.clamp(
            min=-float(log_ratio_clip), max=float(log_ratio_clip)
        )
    ratio = torch.exp(clipped_log_ratio)
    if clip_range is None:
        clipped_ratio = ratio
    else:
        eps = float(clip_range)
        clipped_ratio = ratio.clamp(1.0 - eps, 1.0 + eps)

    weighted_advantage = torch.where(
        advantage >= 0,
        advantage,
        advantage * float(negative_advantage_weight),
    )
    unclipped_obj = ratio * weighted_advantage
    clipped_obj = clipped_ratio * weighted_advantage
    pg_obj = torch.where(
        weighted_advantage >= 0,
        torch.minimum(unclipped_obj, clipped_obj),
        torch.maximum(unclipped_obj, clipped_obj),
    )
    policy_gradient_loss = -_masked_mean(pg_obj, valid)

    loss = policy_gradient_loss + float(kl_coef) * kl_loss
    if reduction == "sum":
        loss = loss * valid.sum().to(dtype=loss.dtype)
    elif reduction not in ("mean", "none"):
        raise ValueError(f"Unsupported GRPO reduction: {reduction}")

    valid_reward = reward[valid].detach()
    valid_advantage = advantage[valid].detach()
    valid_weighted_advantage = weighted_advantage[valid].detach()
    valid_logp_diff = logp_diff[valid].detach()
    valid_ratio = ratio[valid].detach()
    metrics = {
        "grpo_loss_mean": loss.detach(),
        "policy_gradient_loss_mean": policy_gradient_loss.detach(),
        "kl_loss_mean": kl_loss.detach(),
        "kl_scaled_mean": (float(kl_coef) * kl_loss).detach(),
        "reward_mean": valid_reward.mean(),
        "reward_min": valid_reward.min(),
        "reward_max": valid_reward.max(),
        "reward_group_std_mean": reward_std.detach().mean(),
        "reward_std_mean": reward_std.detach().mean(),
        "advantage_mean": valid_advantage.mean(),
        "advantage_std": valid_advantage.std(unbiased=False),
        "advantage_min": valid_advantage.min(),
        "advantage_max": valid_advantage.max(),
        "weighted_advantage_mean": valid_weighted_advantage.mean(),
        "policy_logp_mean": policy_logp[valid].detach().mean(),
        "ref_logp_mean": ref_logp[valid].detach().mean(),
        "logp_diff_mean": valid_logp_diff.mean(),
        "logp_diff_abs_mean": valid_logp_diff.abs().mean(),
        "logp_diff_min": valid_logp_diff.min(),
        "logp_diff_max": valid_logp_diff.max(),
        "ratio_mean": valid_ratio.mean(),
        "ratio_min": valid_ratio.min(),
        "ratio_max": valid_ratio.max(),
        "num_candidates": valid.sum().detach().to(dtype=policy_loss.dtype),
    }
    return loss, metrics
