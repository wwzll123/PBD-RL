from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


METRIC_COLUMNS = [
    "ptx_iptm",
    "ptx_iptm_binder",
    "ptx_plddt",
    "ptx_pred_design_rmsd",
    "ptx_ptm",
    "ptx_ptm_binder",
    "ptx_ptm_target",
]


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _extract_chain_ids_from_pkl_gz(sample_dict: dict[str, Any]) -> list[str]:
    condition = sample_dict["condition"]
    if "bioassembly_dict" not in condition:
        with gzip.open(condition["structure_file"], "rb") as f:
            condition["bioassembly_dict"] = pickle.load(f)
    bioassembly = condition["bioassembly_dict"]
    atom_array = bioassembly["atom_array"]
    return sorted(set(atom_array.chain_id.tolist()))


def _infer_binder_chain(
    chain_ids: list[str], target_chain: str, binder_chain: str | None = None
) -> str:
    if binder_chain is not None:
        if binder_chain not in chain_ids:
            raise ValueError(f"Configured binder_chain '{binder_chain}' not in chains {chain_ids}")
        return binder_chain

    candidates = [chain_id for chain_id in chain_ids if chain_id != target_chain]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"Cannot infer binder chain: only target chain '{target_chain}' exists.")
    raise ValueError(
        f"Cannot infer binder chain from chains {chain_ids}. Pass --binder-chain explicitly."
    )


def _resolve_design_chains(
    chain_ids: list[str],
    target_chain: str | None,
    binder_chain: str | None = None,
) -> tuple[str, str]:
    if target_chain is None and binder_chain is None:
        if "A" in chain_ids and "B" in chain_ids:
            return "A", "B"
        if "A0" in chain_ids and "B0" in chain_ids:
            return "A0", "B0"
        raise ValueError(
            f"Cannot resolve PXDesign output chains from {chain_ids}. "
            "Expected target/binder chains A/B or A0/B0."
        )

    if target_chain is None:
        target_chain = "A0" if binder_chain == "B0" else "A"
    if binder_chain == "B" and "B" not in chain_ids and "B0" in chain_ids:
        binder_chain = "B0"
    elif binder_chain == "B0" and "B0" not in chain_ids and "B" in chain_ids:
        binder_chain = "B"

    resolved_target = target_chain
    if resolved_target not in chain_ids:
        if target_chain == "A" and "A0" in chain_ids:
            resolved_target = "A0"
        elif target_chain == "A0" and "A" in chain_ids:
            resolved_target = "A"
        elif binder_chain is not None and binder_chain in chain_ids:
            candidates = [chain_id for chain_id in chain_ids if chain_id != binder_chain]
            if len(candidates) == 1:
                resolved_target = candidates[0]
            else:
                raise ValueError(
                    f"Configured binder_chain '{binder_chain}' leaves ambiguous "
                    f"target chains {chain_ids}."
                )
        else:
            raise ValueError(
                f"Target chain '{target_chain}' not found in chains {chain_ids}. "
                "PXDesign design outputs should use A/B or A0/B0 by default."
            )

    if binder_chain is None and resolved_target == "A0" and "B0" in chain_ids:
        resolved_binder = "B0"
    elif binder_chain is None and resolved_target == "A" and "B" in chain_ids:
        resolved_binder = "B"
    else:
        resolved_binder = _infer_binder_chain(
            chain_ids, target_chain=resolved_target, binder_chain=binder_chain
        )
    return resolved_target, resolved_binder


def _map_chain_after_processing(chain_id: str, processed_chain_ids: list[str]) -> str:
    if chain_id in processed_chain_ids:
        return chain_id
    suffixed = f"{chain_id}0"
    if suffixed in processed_chain_ids:
        return suffixed
    if len(processed_chain_ids) == 2:
        # PXDesign often renames A/B to A0/B0 after processing.
        base_to_processed = {c.rstrip("0123456789"): c for c in processed_chain_ids}
        if chain_id in base_to_processed:
            return base_to_processed[chain_id]
    raise ValueError(
        f"Cannot map chain '{chain_id}' to processed chains {processed_chain_ids}."
    )


def _atom_signature(atom_array) -> str:
    payload = []
    for chain_id, res_id, atom_name in zip(
        atom_array.chain_id, atom_array.res_id, atom_array.atom_name
    ):
        payload.append(f"{chain_id}:{int(res_id)}:{atom_name}")
    return hashlib.md5("|".join(payload).encode("utf-8")).hexdigest()


def _bioassembly_path_from_cache(input_dict: dict[str, Any], cache_dir: Path) -> Path:
    structure_file = Path(input_dict["condition"]["structure_file"])
    if structure_file.suffixes[-2:] == [".pkl", ".gz"] and structure_file.exists():
        return structure_file
    matches = sorted(cache_dir.glob("*.pkl.gz"))
    if len(matches) != 1:
        raise ValueError(f"Expected one cached bioassembly pkl.gz in {cache_dir}, found {len(matches)}")
    return matches[0]


def _build_feature_and_labels(
    structure_path: Path,
    target_chain: str,
    binder_chain: str | None,
    use_msa: bool,
    cache_dir: Path,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, str, Path]:
    from pxdesign.data.infer_data_pipeline import InferenceDataset
    from pxdesign.utils.infer import convert_to_bioassembly_dict

    input_dict = {
        "name": structure_path.stem,
        "condition": {
            "structure_file": str(structure_path),
            "filter": {
                "chain_id": [],
            },
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    convert_to_bioassembly_dict(input_dict, out_dir=str(cache_dir))
    bioassembly_path = _bioassembly_path_from_cache(input_dict, cache_dir)
    chain_ids = _extract_chain_ids_from_pkl_gz(input_dict)
    if not chain_ids:
        raise ValueError(f"No chains found in {structure_path}")
    _, resolved_binder_chain = _resolve_design_chains(
        chain_ids=chain_ids,
        target_chain=target_chain,
        binder_chain=binder_chain,
    )
    input_dict["condition"]["filter"]["chain_id"] = chain_ids

    ds = InferenceDataset.__new__(InferenceDataset)
    ds.use_msa = use_msa
    processed = ds.process_sample_dict(input_dict)
    data, atom_array, _ = ds.process_one(processed)
    processed_chain_ids = sorted(set(atom_array.chain_id.tolist()))
    resolved_binder_chain = _map_chain_after_processing(
        resolved_binder_chain,
        processed_chain_ids=processed_chain_ids,
    )

    input_feature_dict = data["input_feature_dict"]
    coord = torch.tensor(atom_array.coord, dtype=torch.float32)
    atom_mask = torch.tensor(atom_array.is_resolved.astype(int), dtype=torch.float32)
    binder_mask = torch.tensor(
        (atom_array.chain_id == resolved_binder_chain).astype(int), dtype=torch.float32
    )
    if float(binder_mask.sum()) <= 0.0:
        raise ValueError(
            f"Binder mask is empty for {structure_path}; "
            f"binder_chain={resolved_binder_chain}, processed_chains={processed_chain_ids}"
        )
    return input_feature_dict, coord, atom_mask, binder_mask, _atom_signature(atom_array), bioassembly_path


def _candidate_id(row: dict[str, str]) -> str:
    for key in ("sequence_id", "sample_base", "representative", "global_index"):
        value = row.get(key)
        if value:
            return value
    payload = json.dumps(row, sort_keys=True).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def _reward(row: dict[str, str], args: argparse.Namespace) -> float | None:
    iptm = _float_or_none(row.get("ptx_iptm"))
    ptm_binder = _float_or_none(row.get("ptx_ptm_binder"))
    plddt = _float_or_none(row.get("ptx_plddt"))
    rmsd = _float_or_none(row.get("ptx_pred_design_rmsd"))
    if iptm is None:
        return None

    score = args.w_iptm * iptm
    if ptm_binder is not None:
        score += args.w_ptm_binder * ptm_binder
    if plddt is not None:
        score += args.w_plddt * (plddt / 100.0)
    if rmsd is not None:
        score -= args.w_rmsd_log * math.log1p(max(rmsd, 0.0))
    return float(score)


def _select_ranked(
    ranked: list[dict[str, Any]], max_candidates: int, strategy: str
) -> list[dict[str, Any]]:
    if len(ranked) <= max_candidates:
        return ranked
    if strategy == "top":
        return ranked[:max_candidates]
    if strategy != "stratified":
        raise ValueError(f"Unsupported selection strategy: {strategy}")

    last = len(ranked) - 1
    if max_candidates <= 1:
        return [ranked[0]]
    idxs = sorted({round(i * last / (max_candidates - 1)) for i in range(max_candidates)})
    return [ranked[i] for i in idxs]


def _read_ranked_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    if str(args.structure_path_column) != "complex_pdb_path":
        raise ValueError(
            "GRPO feature construction must use PXDesign-generated complexes from "
            "complex_pdb_path. Do not use pred_pdb_path/Protenix_Pred as the feature source."
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(args.metrics_tsv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"target", args.structure_path_column}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required TSV columns: {missing}")

        for row in reader:
            target = row.get("target", "")
            if not target:
                continue
            path_value = row.get(args.structure_path_column, "")
            if not path_value:
                continue
            path_str = str(Path(path_value))
            if "PXDesign_Gen" not in path_str and "by_target_binders" not in path_str:
                raise ValueError(
                    f"Unexpected feature source for {args.structure_path_column}: {path_value}. "
                    "Expected PXDesign_Gen or legacy by_target_binders."
                )
            score = _reward(row, args)
            if score is None:
                continue
            metrics = {
                name: _float_or_none(row.get(name))
                for name in METRIC_COLUMNS
                if row.get(name) not in (None, "")
            }
            grouped[target].append(
                {
                    "target_key": target,
                    "candidate_id": _candidate_id(row),
                    "structure_path": path_value,
                    "score": score,
                    "metrics": metrics,
                    "source": {
                        "prediction_source": row.get("prediction_source"),
                        "source_shard": row.get("source_shard"),
                        "sample_base": row.get("sample_base"),
                        "representative": row.get("representative"),
                        "design_idx": row.get("design_idx"),
                        "sequence_id": row.get("sequence_id"),
                        "pred_pdb_path": row.get("pred_pdb_path"),
                        "complex_pdb_path": row.get("complex_pdb_path"),
                        "chainB_pdb_path": row.get("chainB_pdb_path"),
                    },
                }
            )

    for rows in grouped.values():
        rows.sort(key=lambda x: x["score"], reverse=True)
        for rank, row in enumerate(rows):
            row["rank"] = rank
    return grouped


def _safe_stem(target: str, candidate_id: str) -> str:
    text = f"{target}__{candidate_id}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def build_groups(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache_bioassembly"
    tensor_dir = out_dir / "tensors"
    feat_dir = tensor_dir / "features"
    coord_dir = tensor_dir / "coords"
    mask_dir = tensor_dir / "masks"
    paths_to_create = [out_dir, cache_dir, coord_dir, mask_dir]
    if bool(args.cache_input_features):
        paths_to_create.append(feat_dir)
    for path in paths_to_create:
        path.mkdir(parents=True, exist_ok=True)

    grouped = _read_ranked_rows(args)
    target_keys = sorted(grouped)
    if args.limit_targets is not None:
        target_keys = target_keys[: max(0, args.limit_targets)]

    group_lines: list[str] = []
    stats = {
        "targets_seen": len(grouped),
        "targets_processed": 0,
        "groups_written": 0,
        "candidates_written": 0,
        "missing_structures": 0,
        "feature_failures": 0,
        "signature_filtered": 0,
    }

    if not bool(args.precompute_tensors):
        for target_key in target_keys:
            selected = _select_ranked(
                grouped[target_key],
                max_candidates=int(args.max_candidates_per_target),
                strategy=str(args.selection_strategy),
            )
            if len(selected) < int(args.min_candidates_per_group):
                continue
            stats["targets_processed"] += 1

            rows = []
            for row in selected:
                structure_path = Path(row["structure_path"])
                if not structure_path.exists():
                    stats["missing_structures"] += 1
                    continue
                rows.append(row)
            if len(rows) < int(args.min_candidates_per_group):
                stats["signature_filtered"] += len(rows)
                continue

            group_rec = {
                "group_id": target_key,
                "target_key": target_key,
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "structure_path": row["structure_path"],
                        "score": float(row["score"]),
                        "rank": int(row["rank"]),
                        "metrics": row["metrics"],
                        "source": row["source"],
                    }
                    for row in rows
                ],
                "meta": {
                    "target_key": target_key,
                    "num_candidates": len(rows),
                    "reward_formula": (
                        f"{args.w_iptm}*ptx_iptm + {args.w_ptm_binder}*ptx_ptm_binder "
                        f"+ {args.w_plddt}*(ptx_plddt/100) "
                        f"- {args.w_rmsd_log}*log1p(ptx_pred_design_rmsd)"
                    ),
                    "structure_path_column": args.structure_path_column,
                    "design_chain_convention": "PXDesign output A/A0 target and B/B0 binder",
                    "feature_cache": "on_the_fly_from_structure_path",
                    "runtime_cache_root": str(out_dir / "runtime_bioassembly"),
                    "target_chain": args.target_chain,
                    "binder_chain": args.binder_chain,
                    "use_msa": bool(args.use_msa),
                    "selection_strategy": args.selection_strategy,
                },
            }
            group_lines.append(json.dumps(group_rec, ensure_ascii=False))
            stats["groups_written"] += 1
            stats["candidates_written"] += len(rows)

        groups_jsonl = out_dir / "train_groups.jsonl"
        with groups_jsonl.open("w", encoding="utf-8") as f:
            for line in group_lines:
                f.write(line + "\n")

        with (out_dir / "build_groups_stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        print(json.dumps(stats, indent=2, sort_keys=True))
        print(f"Wrote {groups_jsonl}")
        return

    for target_key in target_keys:
        selected = _select_ranked(
            grouped[target_key],
            max_candidates=int(args.max_candidates_per_target),
            strategy=str(args.selection_strategy),
        )
        if len(selected) < int(args.min_candidates_per_group):
            continue
        stats["targets_processed"] += 1

        by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
        item_cache: dict[str, dict[str, Path | str]] = {}

        for row in selected:
            structure_path = Path(row["structure_path"])
            if not structure_path.exists():
                stats["missing_structures"] += 1
                continue
            stem = _safe_stem(target_key, row["candidate_id"])
            try:
                input_feat, coord, atom_mask, binder_mask, atom_sig, bioassembly_path = _build_feature_and_labels(
                    structure_path=structure_path,
                    target_chain=args.target_chain,
                    binder_chain=args.binder_chain,
                    use_msa=bool(args.use_msa),
                    cache_dir=cache_dir / stem,
                )
            except Exception as exc:
                stats["feature_failures"] += 1
                print(f"[warn] feature failed for {structure_path}: {exc}")
                continue

            coord_path = coord_dir / f"{stem}_coord.pt"
            atom_mask_path = mask_dir / f"{stem}_atom_mask.pt"
            binder_mask_path = mask_dir / f"{stem}_binder_mask.pt"
            feat_path = None
            if bool(args.cache_input_features):
                feat_path = feat_dir / f"{stem}_input_feature.pt"
                torch.save(input_feat, feat_path)
            torch.save(coord, coord_path)
            torch.save(atom_mask, atom_mask_path)
            torch.save(binder_mask, binder_mask_path)

            item = {
                "bioassembly_path": bioassembly_path,
                "coord_path": coord_path,
                "atom_mask_path": atom_mask_path,
                "binder_mask_path": binder_mask_path,
                "atom_signature": atom_sig,
            }
            if feat_path is not None:
                item["input_feature_path"] = feat_path
            item_cache[row["candidate_id"]] = item
            by_signature[atom_sig].append(row)

        for sig, rows in by_signature.items():
            if len(rows) < int(args.min_candidates_per_group):
                stats["signature_filtered"] += len(rows)
                continue
            rows.sort(key=lambda x: x["score"], reverse=True)
            anchor = item_cache[rows[0]["candidate_id"]]
            group_id = target_key if len(by_signature) == 1 else f"{target_key}__sig_{sig[:8]}"
            group_rec = {
                "group_id": group_id,
                "target_key": target_key,
                "bioassembly_path": str(anchor["bioassembly_path"]),
                "atom_mask_path": str(anchor["atom_mask_path"]),
                "binder_mask_path": str(anchor["binder_mask_path"]),
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "coord_path": str(item_cache[row["candidate_id"]]["coord_path"]),
                        "score": float(row["score"]),
                        "rank": int(row["rank"]),
                        "metrics": row["metrics"],
                        "source": row["source"],
                    }
                    for row in rows
                ],
                "meta": {
                    "target_key": target_key,
                    "atom_signature": sig,
                    "num_candidates": len(rows),
                    "reward_formula": (
                        f"{args.w_iptm}*ptx_iptm + {args.w_ptm_binder}*ptx_ptm_binder "
                        f"+ {args.w_plddt}*(ptx_plddt/100) "
                        f"- {args.w_rmsd_log}*log1p(ptx_pred_design_rmsd)"
                    ),
                    "structure_path_column": args.structure_path_column,
                    "design_chain_convention": "PXDesign output A/A0 target and B/B0 binder",
                    "feature_cache": "input_feature_path" if bool(args.cache_input_features) else "on_the_fly_from_bioassembly_path",
                    "use_msa": bool(args.use_msa),
                    "selection_strategy": args.selection_strategy,
                },
            }
            if "input_feature_path" in anchor:
                group_rec["input_feature_path"] = str(anchor["input_feature_path"])
            group_lines.append(json.dumps(group_rec, ensure_ascii=False))
            stats["groups_written"] += 1
            stats["candidates_written"] += len(rows)

    groups_jsonl = out_dir / "train_groups.jsonl"
    with groups_jsonl.open("w", encoding="utf-8") as f:
        for line in group_lines:
            f.write(line + "\n")

    with (out_dir / "build_groups_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote {groups_jsonl}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline GRPO target groups from Protenix metrics TSV."
    )
    parser.add_argument("--metrics_tsv", required=True)
    parser.add_argument("--out_dir", default="./data/grpo_groups")
    parser.add_argument("--structure_path_column", default="complex_pdb_path")
    parser.add_argument("--max_candidates_per_target", type=int, default=24)
    parser.add_argument("--min_candidates_per_group", type=int, default=4)
    parser.add_argument("--selection_strategy", choices=["stratified", "top"], default="stratified")
    parser.add_argument("--limit_targets", type=int, default=None)
    parser.add_argument("--target_chain", default=None)
    parser.add_argument("--binder_chain", default=None)
    parser.add_argument("--use_msa", action="store_true")
    parser.add_argument(
        "--precompute_tensors",
        action="store_true",
        help=(
            "Precompute bioassembly, coord, atom_mask, and binder_mask tensors during "
            "group building. Disabled by default so full-dataset training can build "
            "these inside dataloader workers."
        ),
    )
    parser.add_argument(
        "--cache_input_features",
        action="store_true",
        help=(
            "Also save full input_feature_dict tensors. Disabled by default because "
            "full-dataset feature tensors are very large; training can featurize from "
            "cached bioassembly .pkl.gz files on the fly."
        ),
    )
    parser.add_argument("--w_iptm", type=float, default=1.0)
    parser.add_argument("--w_ptm_binder", type=float, default=0.2)
    parser.add_argument("--w_plddt", type=float, default=0.2)
    parser.add_argument("--w_rmsd_log", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    build_groups(parse_args())
