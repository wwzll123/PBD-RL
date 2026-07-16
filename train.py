from __future__ import annotations

import copy
import contextlib
import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from pxdesign.utils.infer import get_configs

from PXdpo.data import build_dataloader
from PXdpo.losses import (
    diffusion_dpo_loss,
    diffusion_dpo_loss_multi_candidate,
    diffusion_grpo_loss,
)
from PXdpo.lora import LoRAConfig, inject_lora_linear, mark_only_lora_trainable, merged_lora_state_dict
from PXdpo.modeling import PXDPOModel


def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().float().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_text(name: str, value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return f"{name}=NA"
    return f"{name}={number:.6f}"


def _mask_tensor_by_binder(value: torch.Tensor, binder_mask: torch.Tensor) -> torch.Tensor:
    mask = binder_mask.to(device=value.device, dtype=torch.bool)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    return value.masked_fill(mask, 0)


def _sanitize_binder_condition_features(
    input_feature_dict: dict[str, Any],
    binder_mask: torch.Tensor,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Remove generated binder coordinates from condition features during training."""
    if not enabled:
        return input_feature_dict
    out = dict(input_feature_dict)
    for key in ("condition_coordinate", "condition_coordinate_mask", "condition_atom_mask"):
        value = out.get(key)
        if torch.is_tensor(value):
            out[key] = _mask_tensor_by_binder(value.clone(), binder_mask)
    return out


def _append_metrics_tsv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                key: (
                    f"{number:.8g}"
                    if (number := _as_float(row.get(key))) is not None
                    else row.get(key, "")
                )
                for key in fieldnames
            }
        )




def _trace_value_text(value: Any, limit: int = 1200) -> str:
    if value is None:
        return ""
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        text = ";".join(_trace_value_text(x, limit=limit) for x in value)
    elif isinstance(value, dict):
        text = ",".join(f"{k}:{_trace_value_text(v, limit=limit)}" for k, v in value.items())
    else:
        text = str(value)
    text = text.replace("\t", " ").replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "...TRUNCATED"
    return text


def _trace_rank_event(
    out_dir: Path,
    rank: int,
    epoch: int,
    step: int,
    phase: str,
    batch: dict[str, Any] | None = None,
    *,
    data_format: str = "",
    n_atom: int | None = None,
    num_candidates: int | None = None,
    note: str = "",
) -> None:
    path = out_dir / f"trace_rank{rank}.tsv"
    exists = path.exists()
    target_key = ""
    group_id = ""
    candidate_id = ""
    if batch is not None:
        target_key = _trace_value_text(batch.get("target_key"))
        group_id = _trace_value_text(batch.get("group_id"))
        candidate_id = _trace_value_text(batch.get("candidate_id"))
    fieldnames = [
        "time",
        "rank",
        "epoch",
        "step",
        "phase",
        "data_format",
        "target_key",
        "group_id",
        "candidate_id",
        "n_atom",
        "num_candidates",
        "note",
    ]
    row = {
        "time": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "rank": rank,
        "epoch": epoch,
        "step": step,
        "phase": phase,
        "data_format": data_format,
        "target_key": target_key,
        "group_id": group_id,
        "candidate_id": candidate_id,
        "n_atom": "" if n_atom is None else int(n_atom),
        "num_candidates": "" if num_candidates is None else int(num_candidates),
        "note": _trace_value_text(note, limit=2000),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _init_distributed(cfg: DictConfig) -> tuple[bool, int, int, int]:
    mode = str(cfg.training.get("distributed", "auto")).lower()
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = (mode == "true") or (mode == "auto" and env_world_size > 1)
    if not enabled:
        return False, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA in this training script.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = env_world_size
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=str(cfg.training.ddp_backend))
    return True, local_rank, rank, world_size


def _is_main_process(distributed: bool, rank: int) -> bool:
    return (not distributed) or rank == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


class PXDPOTrainWrapper(torch.nn.Module):
    def __init__(self, model: PXDPOModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, op: str, **kwargs):
        if op == "encode_condition":
            return self.model.encode_condition(**kwargs)
        if op == "diffusion_recon_loss":
            return self.model.diffusion_recon_loss(**kwargs)
        if op == "diffusion_recon_loss_multi_candidates":
            return self.model.diffusion_recon_loss_multi_candidates(**kwargs)
        raise ValueError(f"Unsupported training op: {op}")


def _set_epoch_if_supported(dataloader, epoch: int) -> None:
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _target_group_has_training_signal(
    candidate_score: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    min_reward_std: float,
) -> tuple[bool, str]:
    valid = candidate_valid_mask.to(dtype=torch.bool) & torch.isfinite(candidate_score)
    counts = valid.sum(dim=1)
    if not bool((counts >= 2).all().item()):
        return False, "fewer_than_two_valid_candidates"

    reward = torch.where(valid, candidate_score, torch.zeros_like(candidate_score))
    denom = counts.to(dtype=candidate_score.dtype).clamp_min(1.0).unsqueeze(1)
    mean = reward.sum(dim=1, keepdim=True) / denom
    centered = torch.where(valid, candidate_score - mean, torch.zeros_like(candidate_score))
    std = (centered.square().sum(dim=1) / counts.to(dtype=candidate_score.dtype)).sqrt()
    if not bool((std > min_reward_std).all().item()):
        return False, "zero_or_tiny_reward_std"
    return True, ""


def _format_skip_context(batch: dict[str, Any]) -> str:
    target = batch.get("target_key", ["unknown"])
    group = batch.get("group_id", ["unknown"])
    if isinstance(target, list):
        target = ",".join(str(x) for x in target[:3])
    if isinstance(group, list):
        group = ",".join(str(x) for x in group[:3])
    return f"target={target} group={group}"


def _is_cuda_oom(exc: Exception | None) -> bool:
    if exc is None:
        return False
    exc_name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "outofmemory" in exc_name
        or "out of memory" in message
        or "cuda out of memory" in message
    )


def _sync_step_error(
    distributed: bool,
    device: torch.device,
    exc: Exception | None,
) -> tuple[bool, bool]:
    local_error = exc is not None
    local_non_oom = local_error and not _is_cuda_oom(exc)
    if not distributed:
        return local_error, local_non_oom

    flags = torch.tensor(
        [1 if local_error else 0, 1 if local_non_oom else 0],
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    return bool(flags[0].item()), bool(flags[1].item())


def _all_reduce_gradients(params: list[torch.nn.Parameter]) -> None:
    world_size = dist.get_world_size()
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def _load_checkpoint(
    model: torch.nn.Module,
    load_checkpoint_dir: str,
    model_name: str,
    load_strict: bool,
    device: torch.device,
) -> None:
    ckpt_path = os.path.join(load_checkpoint_dir, f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model"]
    sample_key = next(iter(state.keys()))
    if sample_key.startswith("module."):
        state = {k[len("module.") :]: v for k, v in state.items()}
    model.load_state_dict(state, strict=load_strict)


def _set_trainable_modules(model: torch.nn.Module, trainable_prefixes: list[str]) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            p.requires_grad = True


def _configure_trainable_model(model: torch.nn.Module, cfg: DictConfig) -> str:
    tuning = cfg.model.get("tuning", {})
    method = str(tuning.get("method", "full")).lower()
    if method == "lora":
        target_prefixes = tuple(str(x) for x in tuning.get("target_prefixes", ["diffusion_module"]))
        target_suffixes = tuple(str(x) for x in tuning.get("target_suffixes", [""]))
        lora_cfg = LoRAConfig(
            rank=int(tuning.get("rank", 8)),
            alpha=float(tuning.get("alpha", 16.0)),
            dropout=float(tuning.get("dropout", 0.0)),
            target_prefixes=target_prefixes,
            target_suffixes=target_suffixes,
        )
        replaced = inject_lora_linear(model, lora_cfg)
        if replaced <= 0:
            raise RuntimeError(
                f"LoRA injection found no matching Linear layers. "
                f"target_prefixes={target_prefixes}, target_suffixes={target_suffixes}"
            )
        mark_only_lora_trainable(model)
        return f"lora_linear_layers={replaced}"

    trainable_modules = [str(x) for x in cfg.model.trainable_modules]
    _set_trainable_modules(model, trainable_modules)
    return f"full_trainable_prefixes={trainable_modules}"


def _checkpoint_model_state(model: torch.nn.Module, cfg: DictConfig) -> dict[str, torch.Tensor]:
    tuning = cfg.model.get("tuning", {})
    if str(tuning.get("method", "full")).lower() == "lora" and bool(
        tuning.get("merge_on_save", True)
    ):
        return merged_lora_state_dict(model)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _build_pxdesign_configs(cfg: DictConfig):
    overrides = list(cfg.pxdesign.overrides)
    overrides += [
        "--model_name",
        str(cfg.model.model_name),
        "--load_checkpoint_dir",
        str(to_absolute_path(cfg.model.load_checkpoint_dir)),
        "--load_strict",
        str(cfg.model.load_strict).lower(),
    ]
    return get_configs(overrides)


def _preflight_data_paths(cfg: DictConfig) -> None:
    data_format = str(cfg.data.data_format)
    if data_format == "target_group":
        groups_jsonl = cfg.data.groups_jsonl
        if groups_jsonl is None:
            raise ValueError("data.groups_jsonl is required when data.data_format=target_group.")
        groups_path = Path(to_absolute_path(groups_jsonl))
        if not groups_path.exists():
            raise FileNotFoundError(
                f"Target-group training data not found: {groups_path}\n"
                "Build it first, for example:\n"
                "python PXdpo/make_prefer_pair/build_groups_from_metrics_tsv.py "
                "--metrics_tsv /root/autodl-tmp/binder_grpo_dataset/metadata/"
                "integrated_metrics.flat_paths.tsv "
                "--out_dir /root/autodl-tmp/tmp/grpo_groups "
                "--max_candidates_per_target 24 --min_candidates_per_group 4 "
                "--selection_strategy stratified"
            )
    elif data_format == "pair":
        pairs_jsonl = cfg.data.pairs_jsonl
        if pairs_jsonl is None:
            raise ValueError("data.pairs_jsonl is required when data.data_format=pair.")
        pairs_path = Path(to_absolute_path(pairs_jsonl))
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pair training data not found: {pairs_path}")


def _compute_best_candidate_sft(
    policy_loss: torch.Tensor,
    candidate_rank: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> torch.Tensor:
    masked_rank = torch.where(
        candidate_valid_mask.to(dtype=torch.bool),
        candidate_rank,
        torch.full_like(candidate_rank, 10**9),
    )
    best_rank = masked_rank.min(dim=1, keepdim=True).values
    best_mask = (candidate_rank == best_rank) & candidate_valid_mask.to(dtype=torch.bool)
    best_loss = policy_loss[best_mask]
    if best_loss.numel() == 0:
        raise ValueError("No valid best candidate found for SFT term.")
    return best_loss.mean()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    distributed, local_rank, rank, world_size = _init_distributed(cfg)
    is_main = _is_main_process(distributed, rank)

    torch.manual_seed(int(cfg.experiment.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.experiment.seed))

    device = torch.device(
        f"cuda:{local_rank}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(to_absolute_path(cfg.experiment.output_dir)) / cfg.experiment.run_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))
    if distributed:
        dist.barrier()

    px_cfg = _build_pxdesign_configs(cfg)
    _preflight_data_paths(cfg)

    policy = PXDPOModel(px_cfg).to(device)
    _load_checkpoint(
        model=policy,
        load_checkpoint_dir=to_absolute_path(cfg.model.load_checkpoint_dir),
        model_name=str(cfg.model.model_name),
        load_strict=bool(cfg.model.load_strict),
        device=device,
    )

    trainable_summary = _configure_trainable_model(policy, cfg)
    policy.train()
    if is_main:
        trainable_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in policy.parameters())
        print(
            f"Trainable setup: {trainable_summary}; "
            f"trainable_params={trainable_count:,}/{total_count:,}"
        )

    reference_free = bool(cfg.dpo.reference_free)
    reference = None
    if not reference_free:
        reference = copy.deepcopy(policy).to(device)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad = False

    policy_wrapper = PXDPOTrainWrapper(policy)

    trainable_params = [p for p in policy_wrapper.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check model.trainable_modules.")

    if distributed:
        policy_wrapper = DDP(
            policy_wrapper,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(cfg.training.ddp_find_unused_parameters),
        )
    policy_core = _unwrap_model(policy_wrapper).model

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        betas=tuple(float(x) for x in cfg.training.betas),
    )

    amp_enabled = bool(cfg.training.amp) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled and str(cfg.training.precision) == "fp16")
    amp_dtype = torch.bfloat16 if str(cfg.training.precision) == "bf16" else torch.float16

    dataloader = build_dataloader(
        data_format=str(cfg.data.data_format),
        pairs_jsonl=(
            None if cfg.data.pairs_jsonl is None else to_absolute_path(cfg.data.pairs_jsonl)
        ),
        groups_jsonl=(
            None if cfg.data.groups_jsonl is None else to_absolute_path(cfg.data.groups_jsonl)
        ),
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        group_by_target=bool(cfg.data.group_by_target),
        seed=int(cfg.experiment.seed),
        max_candidates_per_group=(
            None
            if cfg.data.max_candidates_per_group is None
            else int(cfg.data.max_candidates_per_group)
        ),
        min_candidates_per_group=int(cfg.data.min_candidates_per_group),
        max_atoms_per_group=(
            None
            if cfg.data.get("max_atoms_per_group") is None
            else int(cfg.data.max_atoms_per_group)
        ),
        distributed_rank=rank,
        distributed_world_size=world_size,
    )

    global_step = 0
    skipped_steps = 0
    skipped_by_reason: dict[str, int] = {}
    optimizer.zero_grad(set_to_none=True)
    metrics_tsv = out_dir / "train_metrics.tsv"
    metric_fieldnames = [
        "epoch",
        "step",
        "data_format",
        "loss_total",
        "pref_loss",
        "sft_loss",
        "sft_scaled",
        "grad_norm",
        "lr",
        "noise_parallel_steps",
        "skipped_total",
        "policy_loss_mean",
        "policy_loss_min",
        "policy_loss_max",
        "ref_loss_mean",
        "ref_loss_min",
        "ref_loss_max",
        "grpo_loss_mean",
        "policy_gradient_loss_mean",
        "kl_loss_mean",
        "kl_scaled_mean",
        "reward_mean",
        "reward_min",
        "reward_max",
        "reward_group_std_mean",
        "advantage_mean",
        "advantage_std",
        "advantage_min",
        "advantage_max",
        "weighted_advantage_mean",
        "policy_logp_mean",
        "ref_logp_mean",
        "logp_diff_mean",
        "logp_diff_abs_mean",
        "logp_diff_min",
        "logp_diff_max",
        "ratio_mean",
        "ratio_min",
        "ratio_max",
        "num_candidates",
        "policy_delta_mean",
        "logits_mean",
        "dpo_loss_mean",
        "num_preference_pairs",
    ]

    stop_training = False
    max_steps = None if cfg.training.max_steps is None else int(cfg.training.max_steps)

    for epoch in range(int(cfg.training.epochs)):
        _set_epoch_if_supported(dataloader, epoch)
        for batch in dataloader:
            global_step += 1
            _trace_rank_event(
                out_dir,
                rank,
                epoch,
                global_step,
                "batch_loaded",
                batch,
                data_format=str(cfg.data.data_format),
                n_atom=(
                    int(batch["candidate_coord"].shape[2])
                    if "candidate_coord" in batch
                    else int(batch["chosen_coord"].shape[1])
                    if "chosen_coord" in batch
                    else None
                ),
                num_candidates=(
                    int(batch["candidate_coord"].shape[1])
                    if "candidate_coord" in batch
                    else None
                ),
            )

            if str(cfg.data.data_format) == "target_group":
                local_n_atom = int(batch["candidate_coord"].shape[2])
                local_num_candidates = int(batch["candidate_coord"].shape[1])
                local_noise_steps = int(cfg.training.noise_parallel_steps)
                local_group_cost = local_n_atom * local_num_candidates * local_noise_steps
                max_group_cost = int(cfg.training.get("max_group_cost", 0) or 0)
                over_group_cost = max_group_cost > 0 and local_group_cost > max_group_cost
                _trace_rank_event(
                    out_dir,
                    rank,
                    epoch,
                    global_step,
                    "group_cost_checked_local",
                    batch,
                    data_format=str(cfg.data.data_format),
                    n_atom=local_n_atom,
                    num_candidates=local_num_candidates,
                    note=f"group_cost={local_group_cost};max_group_cost={max_group_cost};over={over_group_cost}",
                )
                if distributed:
                    cost_skip_flag = torch.tensor(
                        1 if over_group_cost else 0,
                        device=device,
                        dtype=torch.int64,
                    )
                    dist.all_reduce(cost_skip_flag, op=dist.ReduceOp.MAX)
                    over_group_cost = int(cost_skip_flag.item()) > 0
                if over_group_cost:
                    skipped_steps += 1
                    skip_reason = "max_group_cost"
                    skipped_by_reason[skip_reason] = skipped_by_reason.get(skip_reason, 0) + 1
                    if is_main and (
                        skipped_steps <= 10
                        or skipped_steps % int(cfg.training.log_every_steps) == 0
                    ):
                        print(
                            f"[epoch={epoch} step={global_step}] skip={skip_reason} "
                            f"{_format_skip_context(batch)} "
                            f"group_cost={local_group_cost} max_group_cost={max_group_cost} "
                            f"skipped_total={skipped_steps}"
                        )
                    _trace_rank_event(
                        out_dir,
                        rank,
                        epoch,
                        global_step,
                        "skip_group_cost",
                        batch,
                        data_format=str(cfg.data.data_format),
                        n_atom=local_n_atom,
                        num_candidates=local_num_candidates,
                        note=f"group_cost={local_group_cost};max_group_cost={max_group_cost}",
                    )
                    if max_steps is not None and global_step >= max_steps:
                        stop_training = True
                        break
                    continue

            input_feature_dict = _to_device(batch["input_feature_dict"], device)
            atom_mask = batch["atom_mask"].to(device)
            binder_mask = batch["binder_mask"].to(device)
            input_feature_dict = _sanitize_binder_condition_features(
                input_feature_dict,
                binder_mask,
                enabled=bool(cfg.training.get("mask_binder_condition", True)),
            )
            data_format = str(cfg.data.data_format)

            if data_format == "pair":
                chosen = batch["chosen_coord"].to(device)  # [B, N_atom, 3]
                rejected = batch["rejected_coord"].to(device)
                batch_size = chosen.shape[0]
                n_atom = chosen.shape[1]
            elif data_format == "target_group":
                candidate_coord = batch["candidate_coord"].to(device)  # [B, C, N_atom, 3]
                candidate_score = batch["candidate_score"].to(device)
                candidate_rank = batch["candidate_rank"].to(device)
                candidate_valid_mask = batch["candidate_valid_mask"].to(device)
                has_signal, skip_reason = _target_group_has_training_signal(
                    candidate_score=candidate_score,
                    candidate_valid_mask=candidate_valid_mask,
                    min_reward_std=float(cfg.grpo.min_reward_std) if "grpo" in cfg else 0.0,
                )
                _trace_rank_event(
                    out_dir,
                    rank,
                    epoch,
                    global_step,
                    "signal_checked_local",
                    batch,
                    data_format=data_format,
                    n_atom=int(candidate_coord.shape[2]),
                    num_candidates=int(candidate_coord.shape[1]),
                    note=f"has_signal={has_signal};skip_reason={skip_reason}",
                )
                if distributed:
                    _trace_rank_event(
                        out_dir,
                        rank,
                        epoch,
                        global_step,
                        "signal_allreduce_start",
                        batch,
                        data_format=data_format,
                        n_atom=int(candidate_coord.shape[2]),
                        num_candidates=int(candidate_coord.shape[1]),
                    )
                    skip_flag = torch.tensor(
                        0 if has_signal else 1,
                        device=device,
                        dtype=torch.int64,
                    )
                    dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
                    _trace_rank_event(
                        out_dir,
                        rank,
                        epoch,
                        global_step,
                        "signal_allreduce_done",
                        batch,
                        data_format=data_format,
                        n_atom=int(candidate_coord.shape[2]),
                        num_candidates=int(candidate_coord.shape[1]),
                        note=f"skip_flag={int(skip_flag.item())}",
                    )
                    if int(skip_flag.item()) > 0 and has_signal:
                        has_signal = False
                        skip_reason = "peer_rank_skipped_no_signal"
                if not has_signal:
                    skipped_steps += 1
                    skipped_by_reason[skip_reason] = skipped_by_reason.get(skip_reason, 0) + 1
                    if is_main and (
                        skipped_steps <= 10
                        or skipped_steps % int(cfg.training.log_every_steps) == 0
                    ):
                        print(
                            f"[epoch={epoch} step={global_step}] skip={skip_reason} "
                            f"{_format_skip_context(batch)} skipped_total={skipped_steps}"
                        )
                    _trace_rank_event(
                        out_dir,
                        rank,
                        epoch,
                        global_step,
                        "skip_no_signal",
                        batch,
                        data_format=data_format,
                        n_atom=int(candidate_coord.shape[2]),
                        num_candidates=int(candidate_coord.shape[1]),
                        note=skip_reason,
                    )
                    if max_steps is not None and global_step >= max_steps:
                        stop_training = True
                        break
                    continue
                batch_size = candidate_coord.shape[0]
                n_atom = candidate_coord.shape[2]
            else:
                raise ValueError(f"Unsupported data_format: {data_format}")

            noise_parallel_steps = int(cfg.training.noise_parallel_steps)
            shared_noise = torch.randn(
                batch_size,
                noise_parallel_steps,
                n_atom,
                3,
                device=device,
                dtype=atom_mask.dtype,
            )
            shared_sigma = policy_core._sample_sigmas(
                device=device,
                dtype=atom_mask.dtype,
                batch_size=batch_size,
                num_noise_levels=int(cfg.training.num_noise_levels),
                num_parallel_steps=noise_parallel_steps,
            )

            _trace_rank_event(
                out_dir,
                rank,
                epoch,
                global_step,
                "pre_forward",
                batch,
                data_format=data_format,
                n_atom=n_atom,
                num_candidates=(
                    int(candidate_coord.shape[1])
                    if data_format == "target_group"
                    else None
                ),
            )
            step_exception: Exception | None = None
            grad_norm = None
            try:
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    _trace_rank_event(out_dir, rank, epoch, global_step, "policy_condition_start", batch, data_format=data_format, n_atom=n_atom)
                    policy_condition = policy_wrapper(
                        "encode_condition",
                        input_feature_dict=input_feature_dict,
                    )
                    _trace_rank_event(out_dir, rank, epoch, global_step, "policy_condition_done", batch, data_format=data_format, n_atom=n_atom)

                    with torch.no_grad():
                        if reference_free:
                            reference_condition = None
                        else:
                            assert reference is not None
                            _trace_rank_event(out_dir, rank, epoch, global_step, "reference_condition_start", batch, data_format=data_format, n_atom=n_atom)
                            reference_condition = reference.encode_condition(
                                input_feature_dict=input_feature_dict
                            )
                            _trace_rank_event(out_dir, rank, epoch, global_step, "reference_condition_done", batch, data_format=data_format, n_atom=n_atom)

                    if data_format == "pair":
                        out_pol_good = policy_wrapper(
                            "diffusion_recon_loss",
                            input_feature_dict=input_feature_dict,
                            x0=chosen,
                            atom_mask=atom_mask,
                            binder_mask=binder_mask,
                            num_noise_levels=int(cfg.training.num_noise_levels),
                            num_parallel_steps=noise_parallel_steps,
                            use_binder_weight=bool(cfg.dpo.use_binder_weight),
                            binder_weight=float(cfg.dpo.binder_weight),
                            non_binder_weight=float(cfg.dpo.non_binder_weight),
                            fixed_noise=shared_noise,
                            fixed_sigma=shared_sigma,
                            condition_cache=policy_condition,
                        )
                        out_pol_bad = policy_wrapper(
                            "diffusion_recon_loss",
                            input_feature_dict=input_feature_dict,
                            x0=rejected,
                            atom_mask=atom_mask,
                            binder_mask=binder_mask,
                            num_noise_levels=int(cfg.training.num_noise_levels),
                            num_parallel_steps=noise_parallel_steps,
                            use_binder_weight=bool(cfg.dpo.use_binder_weight),
                            binder_weight=float(cfg.dpo.binder_weight),
                            non_binder_weight=float(cfg.dpo.non_binder_weight),
                            fixed_noise=shared_noise,
                            fixed_sigma=shared_sigma,
                            condition_cache=policy_condition,
                        )
                        _trace_rank_event(out_dir, rank, epoch, global_step, "policy_loss_done", batch, data_format=data_format, n_atom=n_atom, num_candidates=int(candidate_coord.shape[1]))

                        with torch.no_grad():
                            if reference_free:
                                out_ref_good = None
                                out_ref_bad = None
                            else:
                                assert reference is not None
                                out_ref_good = reference.diffusion_recon_loss(
                                    input_feature_dict=input_feature_dict,
                                    x0=chosen,
                                    atom_mask=atom_mask,
                                    binder_mask=binder_mask,
                                    num_noise_levels=int(cfg.training.num_noise_levels),
                                    num_parallel_steps=noise_parallel_steps,
                                    use_binder_weight=bool(cfg.dpo.use_binder_weight),
                                    binder_weight=float(cfg.dpo.binder_weight),
                                    non_binder_weight=float(cfg.dpo.non_binder_weight),
                                    fixed_noise=shared_noise,
                                    fixed_sigma=shared_sigma,
                                    condition_cache=reference_condition,
                                )
                                out_ref_bad = reference.diffusion_recon_loss(
                                    input_feature_dict=input_feature_dict,
                                    x0=rejected,
                                    atom_mask=atom_mask,
                                    binder_mask=binder_mask,
                                    num_noise_levels=int(cfg.training.num_noise_levels),
                                    num_parallel_steps=noise_parallel_steps,
                                    use_binder_weight=bool(cfg.dpo.use_binder_weight),
                                    binder_weight=float(cfg.dpo.binder_weight),
                                    non_binder_weight=float(cfg.dpo.non_binder_weight),
                                    fixed_noise=shared_noise,
                                    fixed_sigma=shared_sigma,
                                    condition_cache=reference_condition,
                                )

                        dpo_loss, dpo_metrics = diffusion_dpo_loss(
                            policy_good=out_pol_good.per_sample_loss,
                            policy_bad=out_pol_bad.per_sample_loss,
                            ref_good=(
                                None if out_ref_good is None else out_ref_good.per_sample_loss
                            ),
                            ref_bad=(
                                None if out_ref_bad is None else out_ref_bad.per_sample_loss
                            ),
                            beta=float(cfg.dpo.beta),
                            reference_free=reference_free,
                            reduction=str(cfg.dpo.loss_reduction),
                        )
                        sft_term = out_pol_good.per_sample_loss.mean()
                    else:
                        _trace_rank_event(out_dir, rank, epoch, global_step, "policy_loss_start", batch, data_format=data_format, n_atom=n_atom, num_candidates=int(candidate_coord.shape[1]))
                        out_pol = policy_wrapper(
                            "diffusion_recon_loss_multi_candidates",
                            input_feature_dict=input_feature_dict,
                            x0=candidate_coord,
                            atom_mask=atom_mask,
                            binder_mask=binder_mask,
                            candidate_valid_mask=candidate_valid_mask,
                            num_noise_levels=int(cfg.training.num_noise_levels),
                            num_parallel_steps=noise_parallel_steps,
                            use_binder_weight=bool(cfg.dpo.use_binder_weight),
                            binder_weight=float(cfg.dpo.binder_weight),
                            non_binder_weight=float(cfg.dpo.non_binder_weight),
                            fixed_noise=shared_noise,
                            fixed_sigma=shared_sigma,
                            condition_cache=policy_condition,
                        )

                        with torch.no_grad():
                            if reference_free:
                                out_ref = None
                            else:
                                assert reference is not None
                                _trace_rank_event(out_dir, rank, epoch, global_step, "reference_loss_start", batch, data_format=data_format, n_atom=n_atom, num_candidates=int(candidate_coord.shape[1]))
                                out_ref = reference.diffusion_recon_loss_multi_candidates(
                                    input_feature_dict=input_feature_dict,
                                    x0=candidate_coord,
                                    atom_mask=atom_mask,
                                    binder_mask=binder_mask,
                                    candidate_valid_mask=candidate_valid_mask,
                                    num_noise_levels=int(cfg.training.num_noise_levels),
                                    num_parallel_steps=noise_parallel_steps,
                                    use_binder_weight=bool(cfg.dpo.use_binder_weight),
                                    binder_weight=float(cfg.dpo.binder_weight),
                                    non_binder_weight=float(cfg.dpo.non_binder_weight),
                                    fixed_noise=shared_noise,
                                    fixed_sigma=shared_sigma,
                                    condition_cache=reference_condition,
                                )
                                _trace_rank_event(out_dir, rank, epoch, global_step, "reference_loss_done", batch, data_format=data_format, n_atom=n_atom, num_candidates=int(candidate_coord.shape[1]))

                        if "grpo" in cfg:
                            _trace_rank_event(out_dir, rank, epoch, global_step, "grpo_loss_start", batch, data_format=data_format, n_atom=n_atom, num_candidates=int(candidate_coord.shape[1]))
                            dpo_loss, dpo_metrics = diffusion_grpo_loss(
                                policy_loss=out_pol.per_sample_loss,
                                ref_loss=(None if out_ref is None else out_ref.per_sample_loss),
                                candidate_score=candidate_score,
                                candidate_valid_mask=candidate_valid_mask,
                                kl_coef=float(cfg.grpo.kl_coef),
                                advantage_eps=float(cfg.grpo.advantage_eps),
                                advantage_clip=(
                                    None
                                    if cfg.grpo.advantage_clip is None
                                    else float(cfg.grpo.advantage_clip)
                                ),
                                reward_clip=(
                                    None
                                    if cfg.grpo.reward_clip is None
                                    else float(cfg.grpo.reward_clip)
                                ),
                                reference_free=reference_free,
                                clip_range=(
                                    None
                                    if cfg.grpo.get("clip_range") is None
                                    else float(cfg.grpo.clip_range)
                                ),
                                log_ratio_clip=(
                                    None
                                    if cfg.grpo.get("log_ratio_clip") is None
                                    else float(cfg.grpo.log_ratio_clip)
                                ),
                                negative_advantage_weight=float(
                                    cfg.grpo.get("negative_advantage_weight", 0.25)
                                ),
                                reduction=str(cfg.dpo.loss_reduction),
                            )
                            valid = candidate_valid_mask.to(dtype=torch.bool)
                            policy_valid = out_pol.per_sample_loss[valid].detach()
                            dpo_metrics["policy_loss_mean"] = policy_valid.mean()
                            dpo_metrics["policy_loss_min"] = policy_valid.min()
                            dpo_metrics["policy_loss_max"] = policy_valid.max()
                            if out_ref is not None:
                                ref_valid = out_ref.per_sample_loss[valid].detach()
                                dpo_metrics["ref_loss_mean"] = ref_valid.mean()
                                dpo_metrics["ref_loss_min"] = ref_valid.min()
                                dpo_metrics["ref_loss_max"] = ref_valid.max()
                        else:
                            dpo_loss, dpo_metrics = diffusion_dpo_loss_multi_candidate(
                                policy_loss=out_pol.per_sample_loss,
                                ref_loss=(None if out_ref is None else out_ref.per_sample_loss),
                                candidate_rank=candidate_rank,
                                candidate_valid_mask=candidate_valid_mask,
                                beta=float(cfg.dpo.beta),
                                reference_free=reference_free,
                                loss_type=str(cfg.dpo.multi_candidate_loss_type),
                                reduction=str(cfg.dpo.loss_reduction),
                            )
                            _trace_rank_event(out_dir, rank, epoch, global_step, "grpo_loss_done", batch, data_format=data_format, n_atom=n_atom, num_candidates=int(candidate_coord.shape[1]))
                        sft_term = _compute_best_candidate_sft(
                            policy_loss=out_pol.per_sample_loss,
                            candidate_rank=candidate_rank,
                            candidate_valid_mask=candidate_valid_mask,
                        )

                    loss = dpo_loss + float(cfg.dpo.sft_coef) * sft_term
                    _trace_rank_event(out_dir, rank, epoch, global_step, "loss_total_done", batch, data_format=data_format, n_atom=n_atom, num_candidates=(int(candidate_coord.shape[1]) if data_format == "target_group" else None), note=f"loss={_as_float(loss)}")
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError(f"Non-finite loss: {loss.detach()}")
            except Exception as exc:
                step_exception = exc
                _trace_rank_event(out_dir, rank, epoch, global_step, "forward_exception", batch, data_format=data_format, n_atom=n_atom, note=f"{type(exc).__name__}:{str(exc)[:500]}")
                if _is_cuda_oom(step_exception):
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            _trace_rank_event(out_dir, rank, epoch, global_step, "forward_error_allreduce_start", batch, data_format=data_format, n_atom=n_atom)
            any_error, any_non_oom_error = _sync_step_error(
                distributed=distributed,
                device=device,
                exc=step_exception,
            )
            _trace_rank_event(out_dir, rank, epoch, global_step, "forward_error_allreduce_done", batch, data_format=data_format, n_atom=n_atom, note=f"any_error={any_error};any_non_oom={any_non_oom_error}")
            if any_error:
                skipped_steps += 1
                skip_reason = (
                    type(step_exception).__name__
                    if step_exception is not None
                    else "peer_rank_forward_error"
                )
                skipped_by_reason[skip_reason] = skipped_by_reason.get(skip_reason, 0) + 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_main:
                    error_text = str(step_exception)[:240] if step_exception is not None else ""
                    print(
                        f"[epoch={epoch} step={global_step}] skip={skip_reason} "
                        f"{_format_skip_context(batch)} error={error_text} "
                        f"skipped_total={skipped_steps}"
                    )
                    if bool(cfg.training.debug_tracebacks) and step_exception is not None:
                        print(
                            "".join(
                                traceback.format_exception(
                                    type(step_exception),
                                    step_exception,
                                    step_exception.__traceback__,
                                )
                            )
                        )
                if distributed and any_non_oom_error:
                    raise RuntimeError(
                        "A non-OOM forward/loss exception occurred on at least one DDP "
                        "rank. Failing fast to avoid rank desynchronization."
                    ) from step_exception
                if max_steps is not None and global_step >= max_steps:
                    stop_training = True
                    break
                continue

            loss_to_backward = loss / int(cfg.training.grad_accum_steps)
            backward_exception: Exception | None = None
            try:
                _trace_rank_event(out_dir, rank, epoch, global_step, "backward_start", batch, data_format=data_format, n_atom=n_atom, num_candidates=(int(candidate_coord.shape[1]) if data_format == "target_group" else None))
                sync_context = (
                    policy_wrapper.no_sync()
                    if distributed
                    else contextlib.nullcontext()
                )
                with sync_context:
                    if scaler.is_enabled():
                        scaler.scale(loss_to_backward).backward()
                    else:
                        loss_to_backward.backward()
                _trace_rank_event(out_dir, rank, epoch, global_step, "backward_done", batch, data_format=data_format, n_atom=n_atom, num_candidates=(int(candidate_coord.shape[1]) if data_format == "target_group" else None))
            except Exception as exc:
                backward_exception = exc
                _trace_rank_event(out_dir, rank, epoch, global_step, "backward_exception", batch, data_format=data_format, n_atom=n_atom, note=f"{type(exc).__name__}:{str(exc)[:500]}")
                if _is_cuda_oom(backward_exception):
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            _trace_rank_event(out_dir, rank, epoch, global_step, "backward_error_allreduce_start", batch, data_format=data_format, n_atom=n_atom)
            any_backward_error, any_backward_non_oom_error = _sync_step_error(
                distributed=distributed,
                device=device,
                exc=backward_exception,
            )
            _trace_rank_event(out_dir, rank, epoch, global_step, "backward_error_allreduce_done", batch, data_format=data_format, n_atom=n_atom, note=f"any_error={any_backward_error};any_non_oom={any_backward_non_oom_error}")
            if any_backward_error:
                skipped_steps += 1
                skip_reason = (
                    type(backward_exception).__name__
                    if backward_exception is not None
                    else "peer_rank_backward_error"
                )
                skipped_by_reason[skip_reason] = skipped_by_reason.get(skip_reason, 0) + 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_main:
                    error_text = (
                        str(backward_exception)[:240]
                        if backward_exception is not None
                        else ""
                    )
                    print(
                        f"[epoch={epoch} step={global_step}] skip={skip_reason} "
                        f"{_format_skip_context(batch)} backward_error={error_text} "
                        f"skipped_total={skipped_steps}"
                    )
                    if bool(cfg.training.debug_tracebacks) and backward_exception is not None:
                        print(
                            "".join(
                                traceback.format_exception(
                                    type(backward_exception),
                                    backward_exception,
                                    backward_exception.__traceback__,
                                )
                            )
                        )
                if distributed and any_backward_non_oom_error:
                    raise RuntimeError(
                        "A non-OOM backward exception occurred on at least one DDP "
                        "rank. Failing fast to avoid rank desynchronization."
                    ) from backward_exception
                if max_steps is not None and global_step >= max_steps:
                    stop_training = True
                    break
                continue

            if global_step % int(cfg.training.grad_accum_steps) == 0:
                if distributed:
                    _trace_rank_event(out_dir, rank, epoch, global_step, "grad_allreduce_start", batch, data_format=data_format, n_atom=n_atom, num_candidates=(int(candidate_coord.shape[1]) if data_format == "target_group" else None))
                    _all_reduce_gradients(trainable_params)
                    _trace_rank_event(out_dir, rank, epoch, global_step, "grad_allreduce_done", batch, data_format=data_format, n_atom=n_atom, num_candidates=(int(candidate_coord.shape[1]) if data_format == "target_group" else None))
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, float(cfg.training.max_grad_norm)
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _trace_rank_event(out_dir, rank, epoch, global_step, "optimizer_step_done", batch, data_format=data_format, n_atom=n_atom, num_candidates=(int(candidate_coord.shape[1]) if data_format == "target_group" else None), note=f"grad_norm={_as_float(grad_norm)}")

            if is_main and global_step % int(cfg.training.log_every_steps) == 0:
                pref_metric = (
                    dpo_metrics["policy_delta_mean"]
                    if "policy_delta_mean" in dpo_metrics
                    else dpo_metrics["advantage_mean"]
                )
                pref_name = "delta" if "policy_delta_mean" in dpo_metrics else "adv"
                log_parts = [
                    f"[epoch={epoch} step={global_step}]",
                    _metric_text("loss", loss),
                    _metric_text("pref", dpo_loss),
                    _metric_text("sft", sft_term),
                    _metric_text("sft_scaled", float(cfg.dpo.sft_coef) * sft_term),
                    _metric_text(pref_name, pref_metric),
                    _metric_text("grad_norm", grad_norm),
                ]
                for key, label in (
                    ("policy_gradient_loss_mean", "pg"),
                    ("kl_loss_mean", "kl"),
                    ("kl_scaled_mean", "kl_scaled"),
                    ("reward_mean", "reward"),
                    ("reward_group_std_mean", "reward_std"),
                    ("ratio_mean", "ratio"),
                    ("ratio_max", "ratio_max"),
                    ("logp_diff_abs_mean", "|dlogp|"),
                    ("policy_loss_mean", "pol_loss"),
                    ("ref_loss_mean", "ref_loss"),
                    ("num_candidates", "n_cand"),
                ):
                    if key in dpo_metrics:
                        log_parts.append(_metric_text(label, dpo_metrics[key]))
                log_parts.extend(
                    [
                        f"K={noise_parallel_steps}",
                        f"skipped={skipped_steps}",
                    ]
                )
                print(
                    " ".join(log_parts)
                )
                metrics_row = {
                    "epoch": epoch,
                    "step": global_step,
                    "data_format": data_format,
                    "loss_total": loss,
                    "pref_loss": dpo_loss,
                    "sft_loss": sft_term,
                    "sft_scaled": float(cfg.dpo.sft_coef) * sft_term,
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                    "noise_parallel_steps": noise_parallel_steps,
                    "skipped_total": skipped_steps,
                }
                metrics_row.update(dpo_metrics)
                _append_metrics_tsv(metrics_tsv, metrics_row, metric_fieldnames)

            if is_main and global_step % int(cfg.training.save_every_steps) == 0:
                ckpt_path = out_dir / f"step_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model": _checkpoint_model_state(policy_core, cfg),
                        "optimizer": optimizer.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_path,
                )
            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break
        if stop_training:
            break

    if is_main:
        final_ckpt = out_dir / "final.pt"
        torch.save(
            {
                "step": global_step,
                "model": _checkpoint_model_state(policy_core, cfg),
                "optimizer": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            final_ckpt,
        )
        print(f"Training finished. Final checkpoint: {final_ckpt}")
        if skipped_by_reason:
            print(f"Skipped batches by reason: {skipped_by_reason}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
