from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from networkx.classes import non_edges
import torch

from pxdesign.data.infer_data_pipeline import InferenceDataset
from pxdesign.model.pxdesign import ProtenixDesign
from pxdesign.runner.dumper import save_structure_cif
from pxdesign.utils.infer import get_configs


def load_pxdesign_model(
    model_name: str,
    load_checkpoint_dir: str,
    device: torch.device,
    dtype: str = "bf16",
    n_step: int = 400,
    n_sample: int = 1,
    eta_type: str = "const",
    eta_min: float = 2.5,
    eta_max: float = 2.5,
    use_fast_ln: bool = True,
    load_strict: bool = True,
) -> ProtenixDesign:
    """
    Build and load PXDesign model without touching pxdesign source.
    """
    argv = [
        "--model_name",
        model_name,
        "--load_checkpoint_dir",
        load_checkpoint_dir,
        "--load_strict",
        str(load_strict).lower(),
        "--dtype",
        dtype,
        "--N_step",
        str(n_step),
        "--N_sample",
        str(n_sample),
        "--eta_type",
        eta_type,
        "--eta_min",
        str(eta_min),
        "--eta_max",
        str(eta_max),
        "--use_fast_ln",
        str(use_fast_ln).lower(),
        # only to satisfy required arg in parser; not used here.
        "--input_json_path",
        "/tmp/dummy.json",
    ]
    configs = get_configs(argv)
    model = ProtenixDesign(configs).to(device)

    ckpt_path = os.path.join(load_checkpoint_dir, f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"]
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith("module."):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=load_strict)
    model.eval()
    return model


def featurize_task_dict(
    task_dict: dict[str, Any],
    use_msa: bool = True,
) -> tuple[dict[str, Any], Any]:
    """
    Reuse PXDesign inference featurization for a single task dict.
    Returns:
      data: includes input_feature_dict
      atom_array: original atom array
    """
    ds = InferenceDataset.__new__(InferenceDataset)
    ds.use_msa = use_msa
    processed = ds.process_sample_dict(task_dict)
    data, atom_array, _ = ds.process_one(processed)
    return data, atom_array


def save_partial_samples_to_cif(
    coords: torch.Tensor,  # [N_sample, N_atom, 3]
    atom_array,
    entity_poly_type: dict[str, str],
    out_dir: str | Path,
    sample_prefix: str,
) -> list[Path]:
    """
    Save N_sample predicted coordinates to CIF files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for i in range(coords.shape[0]):
        cif_path = out_dir / f"{sample_prefix}_sample_{i}.cif"
        save_structure_cif(
            atom_array=atom_array,
            pred_coordinate=coords[i].detach().cpu(),
            output_fpath=str(cif_path),
            entity_poly_type=entity_poly_type,
            pdb_id=f"{sample_prefix}_sample_{i}",
        )
        saved_paths.append(cif_path)
    return saved_paths


def default_update_atom_mask(input_feature_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Default update mask = design atoms.
    In PXDesign features:
      condition_atom_mask == 1 means conditional/fixed atom
      so design atom mask is its complement.
    """
    if "condition_atom_mask" not in input_feature_dict:
        raise KeyError("condition_atom_mask not found in input_feature_dict")
    cond = input_feature_dict["condition_atom_mask"].bool()
    return ~cond


def validate_fixed_regions(
    atom_array,
    update_atom_mask: torch.Tensor,
    unique_ca_threshold: int = 5,
) -> None:
    """
    Guard against clamping generated-chain placeholder coordinates.

    For PXDesign generation chains (res_name = xpb), the initial coordinates can be
    collapsed placeholders. If those atoms are fixed in partial diffusion, severe atom
    overlap is expected.
    """
    fixed = ~update_atom_mask.cpu().numpy().astype(bool)
    chain_ids = sorted(set(atom_array.chain_id.tolist()))
    for chain_id in chain_ids:
        chain_mask = atom_array.chain_id == chain_id
        chain_fixed = chain_mask & fixed
        if not chain_fixed.any():
            continue
        # Heuristic for generated binder chain in PXDesign inference pipeline.
        chain_res_names = atom_array.res_name[chain_mask]
        is_generated_chain = len(chain_res_names) > 0 and np.all(chain_res_names == "xpb")
        if not is_generated_chain:
            continue
        ca_mask = chain_fixed & (atom_array.atom_name == "CA")
        if not ca_mask.any():
            continue
        ca_coords = np.round(atom_array.coord[ca_mask], 3)
        n_unique = len(np.unique(ca_coords, axis=0))
        if n_unique < unique_ca_threshold:
            raise ValueError(
                f"Fixed region includes generated chain {chain_id} with degenerate "
                f"placeholder coordinates (unique CA={n_unique}). "
                "This will cause atom overlap. "
                "Use full-chain update for that generated chain, or provide an external "
                "scaffold coordinate set as fixed_atom_coords."
            )


def detect_generated_chain_id(atom_array) -> str:
    """
    Detect PXDesign generated protein chain by residue name marker ``xpb``.
    """
    chain_ids = sorted(set(atom_array.chain_id.tolist()))
    candidates: list[tuple[int, str]] = []
    for chain_id in chain_ids:
        chain_mask = atom_array.chain_id == chain_id
        chain_res_name = atom_array.res_name[chain_mask]
        if len(chain_res_name) > 0 and np.all(chain_res_name == "xpb"):
            n_res = len(np.unique(atom_array.res_id[chain_mask]))
            candidates.append((n_res, chain_id))
    if not candidates:
        raise ValueError(
            "Cannot find generated chain (res_name='xpb'). "
            "Please verify input task or set binder chain manually."
        )
    # Prefer the longest generated chain when multiple exist.
    candidates.sort(reverse=True)
    return candidates[0][1]


def detect_protein_chain_ids(atom_array) -> list[str]:
    """
    Detect protein chains in featurized atom_array.
    """
    chain_ids = sorted(set(atom_array.chain_id.tolist()))
    protein_chains: list[str] = []
    for chain_id in chain_ids:
        chain_mask = atom_array.chain_id == chain_id
        if "is_protein" in atom_array._annot and np.any(atom_array.is_protein[chain_mask]):
            protein_chains.append(chain_id)
            continue
        # fallback by residue marker
        res_names = np.unique(atom_array.res_name[chain_mask])
        if len(res_names) > 0 and not np.all(res_names == "xpb"):
            protein_chains.append(chain_id)
    return protein_chains


@torch.no_grad()
def sample_diffusion_partial(
    model: ProtenixDesign,
    input_feature_dict: dict[str, torch.Tensor],
    update_atom_mask: Optional[torch.Tensor] = None,  # [N_atom], True means update
    fixed_atom_coords: Optional[torch.Tensor] = None,  # [N_atom, 3]
    fixed_atom_mask: Optional[torch.Tensor] = None,  # [N_atom], True means clamp
    n_sample: Optional[int] = None,
    n_step: Optional[int] = None,
    gamma0: Optional[float] = None,
    gamma_min: Optional[float] = None,
    noise_scale_lambda: Optional[float] = None,
    step_scale_eta: Optional[float | dict[str, float | str]] = None,
) -> torch.Tensor:
    """
    Partial diffusion sampler with hard coordinate clamping on masked atoms.
    This implements "only update masked atoms each denoise step".
    """
    cfg = model.configs
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32

    feat = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_feature_dict.items()}
    s_inputs, s, z = model.get_condition_embedding(feat, chunk_size=cfg.infer_setting.chunk_size)

    _n_sample = int(n_sample if n_sample is not None else cfg.sample_diffusion["N_sample"])
    _n_step = int(n_step if n_step is not None else cfg.sample_diffusion["N_step"])
    _gamma0 = float(gamma0 if gamma0 is not None else cfg.sample_diffusion["gamma0"])
    _gamma_min = float(gamma_min if gamma_min is not None else cfg.sample_diffusion["gamma_min"])
    _noise_scale_lambda = float(
        noise_scale_lambda
        if noise_scale_lambda is not None
        else cfg.sample_diffusion["noise_scale_lambda"]
    )
    _eta = step_scale_eta if step_scale_eta is not None else cfg.sample_diffusion["eta_schedule"]

    noise_schedule = model.inference_noise_scheduler(N_step=_n_step, device=device, dtype=dtype)
    n_atom = feat["atom_to_token_idx"].shape[-1]

    if update_atom_mask is None:
        update_atom_mask = default_update_atom_mask(feat)
    update_atom_mask = update_atom_mask.to(device).bool()

    if fixed_atom_mask is None:
        fixed_atom_mask = ~update_atom_mask
    fixed_atom_mask = fixed_atom_mask.to(device).bool()

    if fixed_atom_coords is None:
        if "condition_coordinate" in feat and "condition_coordinate_mask" in feat:
            cond_xyz = feat["condition_coordinate"].to(device)
            cond_m = feat["condition_coordinate_mask"].to(device).bool()
            fixed_atom_coords = cond_xyz
            fixed_atom_mask = fixed_atom_mask & cond_m
        else:
            raise KeyError(
                "Cannot determine fixed_atom_coords from input features. "
                "When 'condition_coordinate' is absent, please pass "
                "'fixed_atom_coords' explicitly (e.g. from atom_array.coord)."
            )
    else:
        fixed_atom_coords = fixed_atom_coords.to(device)

    # [N_sample, N_atom, 3]
    x = noise_schedule[0] * torch.randn((_n_sample, n_atom, 3), device=device, dtype=dtype)
    if fixed_atom_mask.any():
        x[:, fixed_atom_mask] = fixed_atom_coords[fixed_atom_mask].to(x.dtype)

    total_t = len(noise_schedule)
    for step_t, (c_tau_last, c_tau) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
        gamma = _gamma0 if c_tau > _gamma_min else 0.0
        t_hat = c_tau_last * (1.0 + gamma)
        delta = torch.sqrt(t_hat**2 - c_tau_last**2)
        x_noisy = x + _noise_scale_lambda * delta * torch.randn_like(x)

        t_hat_vec = t_hat.reshape(1).expand(_n_sample).to(dtype)
        x_denoised = model.diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat_vec,
            input_feature_dict=feat,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            chunk_size=None,
            inplace_safe=True,
        )
        denoised_over_sigma = (x_noisy - x_denoised) / t_hat_vec[:, None, None]
        dt = c_tau - t_hat
        if isinstance(_eta, (float, int)):
            eta = float(_eta)
        elif isinstance(_eta, dict):
            et = str(_eta.get("type", "const"))
            emn = float(_eta.get("min", 1.5))
            emx = float(_eta.get("max", emn))
            if et == "const":
                eta = emn
            elif et == "linear":
                eta = emn + (emx - emn) * (step_t / total_t)
            else:
                # keep behavior simple and deterministic
                eta = emn
        else:
            eta = 1.5

        x_next = x_noisy + eta * dt * denoised_over_sigma
        # hard clamp: only masked atoms are updated
        x = torch.where(update_atom_mask[None, :, None], x_next, x)
        if fixed_atom_mask.any():
            x[:, fixed_atom_mask] = fixed_atom_coords[fixed_atom_mask].to(x.dtype)

    return x  # [N_sample, N_atom, 3]


if __name__ == "__main__":

    device = torch.device("cuda:0")
    model = load_pxdesign_model(
        model_name="pxdesign_v0.1.0",
        load_checkpoint_dir="/root/PXDesign-main/release_data/checkpoint",
        device=device,
    )

    task = {
        "name": "5sa3_binder_generation",
        "condition": {
            "structure_file": "/root/PXDesign/PBD-RL/example_bio_gz/5sa3.pkl.gz",
            # Binder generation only: keep target chain.
            "filter": {"chain_id": ["A0"], "crop": {"A0": "1-100"}},
            "msa": {},
        },
        # Pure binder generation (single designed protein chain).
        "generation": [{"type": "protein", "length": 132, "count": 1}],
    }

    data, atom_array = featurize_task_dict(task, use_msa=False)
    feat = data["input_feature_dict"]

    # Binder generation only: update all design atoms.
    update_atom_mask = None
    fixed_atom_coords = torch.from_numpy(atom_array.coord).to(torch.float32)
    coords = sample_diffusion_partial(
        model,
        feat,
        update_atom_mask=update_atom_mask,
        fixed_atom_coords=fixed_atom_coords,
        n_sample=8,
        n_step=20,
    )
    print(coords.shape)  # [N_sample, N_atom, 3]

    out_dir = Path("/root/PXDesign/PBD-RL/partial_outputs") / task["name"]
    saved = save_partial_samples_to_cif(
        coords=coords,
        atom_array=atom_array,
        entity_poly_type=data["entity_poly_type"],
        out_dir=out_dir,
        sample_prefix=task["name"],
    )
    print(f"Saved {len(saved)} CIF files to: {out_dir}")