from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from pxdesign.data.infer_data_pipeline import InferenceDataset
from pxdesign.utils.infer import convert_to_bioassembly_dict


def _read_scores(score_json: Path) -> dict[str, dict[str, float]]:
    with score_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict[str, float]] = {}
    for k, v in raw.items():
        out[k] = {
            "pTM": float(v["pTM"]),
            "ipTM": float(v["ipTM"]),
        }
    return out


def _group_keys_by_target(score_keys: list[str]) -> dict[str, list[str]]:
    """
    Example:
      1a0a_B_0 -> group key 1a0a_B
      1a0a_B_1 -> group key 1a0a_B
    """
    groups: dict[str, list[str]] = {}
    for key in score_keys:
        parts = key.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(
                f"Score key '{key}' does not match expected pattern <target>_<index>."
            )
        gk = parts[0]
        groups.setdefault(gk, []).append(key)
    return groups


def _extract_chain_ids_from_pkl_gz(sample_dict: dict[str, Any]) -> list[str]:
    bioassembly = sample_dict["condition"]["bioassembly_dict"]
    atom_array = bioassembly["atom_array"]
    return sorted(set(atom_array.chain_id.tolist()))


def _infer_binder_chain(
    chain_ids: list[str], target_chain: str, binder_chain: str | None = None
) -> str:
    if binder_chain is not None:
        if binder_chain not in chain_ids:
            raise ValueError(
                f"Configured binder_chain '{binder_chain}' not in chains {chain_ids}"
            )
        return binder_chain

    cand = [c for c in chain_ids if c != target_chain]
    if len(cand) == 1:
        return cand[0]
    if len(cand) == 0:
        raise ValueError(
            f"Cannot infer binder chain: only target chain '{target_chain}' exists."
        )
    raise ValueError(
        f"Cannot infer binder chain from chains {chain_ids}. "
        "Please set make_pairs.binder_chain explicitly."
    )


def _resolve_design_chains(
    chain_ids: list[str],
    target_chain: str,
    binder_chain: str | None = None,
) -> tuple[str, str]:
    resolved_target = target_chain
    if resolved_target not in chain_ids:
        # Protenix-designed CIFs typically rename chains to A0/B0.
        if "A0" in chain_ids and "B0" in chain_ids:
            resolved_target = "A0"
        elif binder_chain is not None and binder_chain in chain_ids:
            cand = [c for c in chain_ids if c != binder_chain]
            if len(cand) == 1:
                resolved_target = cand[0]
            else:
                raise ValueError(
                    f"Configured binder_chain '{binder_chain}' leaves ambiguous target "
                    f"chains {chain_ids}. Please set make_pairs.target_chain explicitly."
                )
        else:
            raise ValueError(
                f"Target chain '{target_chain}' not found in chains {chain_ids}. "
                "For Protenix-designed CIFs, set make_pairs.target_chain=A0 and "
                "make_pairs.binder_chain=B0 if needed."
            )

    if binder_chain is None and resolved_target == "A0" and "B0" in chain_ids:
        resolved_binder = "B0"
    else:
        resolved_binder = _infer_binder_chain(
            chain_ids, target_chain=resolved_target, binder_chain=binder_chain
        )
    return resolved_target, resolved_binder


def _atom_signature(atom_array) -> str:
    payload = []
    for c, r, n in zip(atom_array.chain_id, atom_array.res_id, atom_array.atom_name):
        payload.append(f"{c}:{int(r)}:{n}")
    txt = "|".join(payload).encode("utf-8")
    return hashlib.md5(txt).hexdigest()


def _build_feature_and_labels(
    cif_path: Path,
    target_chain: str,
    binder_chain: str | None,
    use_msa: bool,
    cache_dir: Path,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Returns:
      input_feature_dict, full_coord[N_atom,3], atom_mask[N_atom], binder_mask[N_atom]
    """
    # Reuse PXDesign conversion utility to generate pkl.gz bioassembly.
    input_dict = {
        "name": cif_path.stem,
        "condition": {
            "structure_file": str(cif_path),
            "filter": {
                # Keep all chains in the complex as condition context.
                "chain_id": [],
            },
        },
    }
    convert_to_bioassembly_dict(input_dict, out_dir=str(cache_dir))
    chain_ids = _extract_chain_ids_from_pkl_gz(input_dict)
    if not chain_ids:
        raise ValueError(f"No chains found in {cif_path}")
    resolved_target_chain, resolved_binder_chain = _resolve_design_chains(
        chain_ids=chain_ids,
        target_chain=target_chain,
        binder_chain=binder_chain,
    )
    # Fill filter chain ids after conversion for process_sample_dict validity checks.
    input_dict["condition"]["filter"]["chain_id"] = chain_ids

    ds = InferenceDataset.__new__(InferenceDataset)
    ds.use_msa = use_msa
    processed = ds.process_sample_dict(input_dict)
    data, atom_array, _ = ds.process_one(processed)

    input_feature_dict = data["input_feature_dict"]
    coord = torch.tensor(atom_array.coord, dtype=torch.float32)
    atom_mask = torch.tensor(atom_array.is_resolved.astype(int), dtype=torch.float32)

    binder_mask_np = (atom_array.chain_id == resolved_binder_chain).astype(int)
    binder_mask = torch.tensor(binder_mask_np, dtype=torch.float32)

    signature = _atom_signature(atom_array)
    return input_feature_dict, coord, atom_mask, binder_mask, signature


def _make_pairs(sorted_items: list[tuple[str, float]], pairs_per_target: int) -> list[tuple[str, str]]:
    """
    Build preference pairs from ranked candidates.
    chosen from top half, rejected from bottom half.
    """
    n = len(sorted_items)
    if n < 2:
        return []
    top = [x[0] for x in sorted_items[: max(1, n // 2)]]
    bottom = [x[0] for x in sorted_items[max(1, n // 2) :]]
    if not bottom:
        return []

    pairs: list[tuple[str, str]] = []
    max_pairs = min(pairs_per_target, len(top) * len(bottom))
    # Enumerate deterministic all candidates then truncate.
    for c in top:
        for r in bottom:
            pairs.append((c, r))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _make_group_candidates(
    ranked: list[tuple[str, float]],
    item_cache: dict[str, dict[str, Path | str]],
    max_candidates_per_target: int,
) -> list[tuple[int, str, float]]:
    if not ranked:
        return []
    anchor_signature = str(item_cache[ranked[0][0]]["atom_signature"])
    group_candidates: list[tuple[int, str, float]] = []
    for rank, (item_key, score) in enumerate(ranked):
        if str(item_cache[item_key]["atom_signature"]) != anchor_signature:
            continue
        group_candidates.append((rank, item_key, score))
        if len(group_candidates) >= max_candidates_per_target:
            break
    return group_candidates


@hydra.main(version_base=None, config_path="../configs", config_name="make_pairs")
def main(cfg: DictConfig) -> None:
    random.seed(int(cfg.experiment.seed))

    complex_dir = Path(to_absolute_path(cfg.input.complex_dir))
    score_json = Path(to_absolute_path(cfg.input.score_json))
    out_dir = Path(to_absolute_path(cfg.output.out_dir))
    cache_dir = out_dir / "cache_bioassembly"
    tensor_dir = out_dir / "tensors"
    feat_dir = tensor_dir / "features"
    coord_dir = tensor_dir / "coords"
    mask_dir = tensor_dir / "masks"
    for d in [out_dir, cache_dir, tensor_dir, feat_dir, coord_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with (out_dir / "resolved_make_pairs_config.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    scores = _read_scores(score_json)
    grouped = _group_keys_by_target(list(scores.keys()))

    # cache sample tensors by item key, avoid repeated cif parsing.
    item_cache: dict[str, dict[str, Path | str]] = {}
    pair_lines: list[str] = []
    group_lines: list[str] = []

    for target_key, item_keys in grouped.items():
        ranked = []
        for k in item_keys:
            s = scores[k]
            score = float(cfg.pairing.w_ptm) * s["pTM"] + float(cfg.pairing.w_iptm) * s["ipTM"]
            ranked.append((k, score))
        ranked.sort(key=lambda x: x[1], reverse=True)

        pairs = _make_pairs(ranked, pairs_per_target=int(cfg.pairing.pairs_per_target))
        if not pairs:
            continue

        # Parse target chain from target key: <pdbid>_<targetChain>
        parts = target_key.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Target key format error: {target_key}")
        target_chain = (
            str(cfg.make_pairs.target_chain)
            if cfg.make_pairs.target_chain
            else parts[1]
        )

        for k, _ in ranked:
            if k in item_cache:
                continue
            cif_path = complex_dir / f"{k}.cif"
            if not cif_path.exists():
                raise FileNotFoundError(f"CIF not found: {cif_path}")

            input_feat, coord, atom_mask, binder_mask, atom_sig = _build_feature_and_labels(
                cif_path=cif_path,
                target_chain=target_chain,
                binder_chain=(
                    None
                    if not cfg.make_pairs.binder_chain
                    else str(cfg.make_pairs.binder_chain)
                ),
                use_msa=bool(cfg.make_pairs.use_msa),
                cache_dir=cache_dir,
            )

            feat_path = feat_dir / f"{k}_input_feature.pt"
            coord_path = coord_dir / f"{k}_coord.pt"
            atom_mask_path = mask_dir / f"{k}_atom_mask.pt"
            binder_mask_path = mask_dir / f"{k}_binder_mask.pt"
            torch.save(input_feat, feat_path)
            torch.save(coord, coord_path)
            torch.save(atom_mask, atom_mask_path)
            torch.save(binder_mask, binder_mask_path)
            item_cache[k] = {
                "input_feature_path": feat_path,
                "coord_path": coord_path,
                "atom_mask_path": atom_mask_path,
                "binder_mask_path": binder_mask_path,
                "atom_signature": atom_sig,
            }

        group_candidates = _make_group_candidates(
            ranked=ranked,
            item_cache=item_cache,
            max_candidates_per_target=int(cfg.pairing.max_candidates_per_target),
        )
        if len(group_candidates) >= int(cfg.pairing.min_candidates_per_target):
            anchor_key = group_candidates[0][1]
            group_rec = {
                "group_id": target_key,
                "target_key": target_key,
                "input_feature_path": str(item_cache[anchor_key]["input_feature_path"]),
                "atom_mask_path": str(item_cache[anchor_key]["atom_mask_path"]),
                "binder_mask_path": str(item_cache[anchor_key]["binder_mask_path"]),
                "candidates": [
                    {
                        "candidate_id": candidate_key,
                        "coord_path": str(item_cache[candidate_key]["coord_path"]),
                        "score": float(score),
                        "rank": int(rank),
                        "metrics": scores[candidate_key],
                    }
                    for rank, candidate_key, score in group_candidates
                ],
                "meta": {
                    "target_key": target_key,
                    "num_candidates": len(group_candidates),
                    "candidate_ids": [candidate_key for _, candidate_key, _ in group_candidates],
                },
            }
            group_lines.append(json.dumps(group_rec, ensure_ascii=False))

        for i, (chosen_key, rejected_key) in enumerate(pairs):
            if (
                item_cache[chosen_key]["atom_signature"]
                != item_cache[rejected_key]["atom_signature"]
            ):
                # Skip misaligned atom ordering/layout between chosen and rejected.
                continue
            rec = {
                "pair_id": f"{target_key}_{i}",
                "target_key": target_key,
                "input_feature_path": str(item_cache[chosen_key]["input_feature_path"]),
                "chosen_coord_path": str(item_cache[chosen_key]["coord_path"]),
                "rejected_coord_path": str(item_cache[rejected_key]["coord_path"]),
                "atom_mask_path": str(item_cache[chosen_key]["atom_mask_path"]),
                "binder_mask_path": str(item_cache[chosen_key]["binder_mask_path"]),
                "meta": {
                    "target_key": target_key,
                    "chosen": chosen_key,
                    "rejected": rejected_key,
                    "chosen_scores": scores[chosen_key],
                    "rejected_scores": scores[rejected_key],
                },
            }
            pair_lines.append(json.dumps(rec, ensure_ascii=False))

    pairs_jsonl = out_dir / "train_pairs.jsonl"
    with pairs_jsonl.open("w", encoding="utf-8") as f:
        for line in pair_lines:
            f.write(line + "\n")

    groups_jsonl = out_dir / "train_groups.jsonl"
    with groups_jsonl.open("w", encoding="utf-8") as f:
        for line in group_lines:
            f.write(line + "\n")

    print(f"Generated {len(pair_lines)} pairs.")
    print(f"Pairs file: {pairs_jsonl}")
    print(f"Generated {len(group_lines)} target groups.")
    print(f"Target groups file: {groups_jsonl}")
    print(f"Tensor cache dir: {tensor_dir}")


if __name__ == "__main__":
    main()

