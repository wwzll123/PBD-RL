import json
import random
import gzip
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def _load_tensor(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    raise TypeError(f"Expected torch.Tensor at {path}, got {type(obj)}")


def _load_feature_dict(path: str | Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected dict at {path}, got {type(obj)}")


def _count_structure_atoms(path: str | Path) -> int | None:
    path = Path(path)
    if path.suffix == ".pdb":
        count = 0
        with path.open("rb") as f:
            for line in f:
                if line.startswith(b"ATOM") or line.startswith(b"HETATM"):
                    count += 1
        return count
    return None


def _chain_ids_from_bioassembly_pkl(path: str | Path) -> list[str]:
    with gzip.open(path, "rb") as f:
        bioassembly = pickle.load(f)
    atom_array = bioassembly["atom_array"]
    return sorted(set(atom_array.chain_id.tolist()))


def _build_feature_dict_from_bioassembly(path: str | Path, use_msa: bool = False) -> dict[str, Any]:
    from pxdesign.data.infer_data_pipeline import InferenceDataset

    bioassembly_path = Path(path)
    input_dict = {
        "name": bioassembly_path.stem.replace(".pkl", ""),
        "condition": {
            "structure_file": str(bioassembly_path),
            "filter": {
                "chain_id": _chain_ids_from_bioassembly_pkl(bioassembly_path),
            },
        },
    }
    ds = InferenceDataset.__new__(InferenceDataset)
    ds.use_msa = use_msa
    processed = ds.process_sample_dict(input_dict)
    data, _, _ = ds.process_one(processed)
    return data["input_feature_dict"]


def _safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _build_candidate_features_and_labels(
    candidate: dict[str, Any],
    *,
    cache_root: Path,
    target_chain: str | None = None,
    binder_chain: str | None = None,
    use_msa: bool = False,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, str]:
    from pxdesign.data.infer_data_pipeline import InferenceDataset
    from pxdesign.utils.infer import convert_to_bioassembly_dict

    from PXdpo.make_prefer_pair.build_groups_from_metrics_tsv import (
        _atom_signature,
        _bioassembly_path_from_cache,
        _map_chain_after_processing,
        _resolve_design_chains,
    )

    source_path = candidate.get("bioassembly_path") or candidate.get("structure_path")
    if not source_path:
        raise KeyError("Candidate must contain structure_path or bioassembly_path.")

    source_path = Path(source_path)
    input_dict = {
        "name": source_path.stem.replace(".pkl", ""),
        "condition": {
            "structure_file": str(source_path),
            "filter": {
                "chain_id": [],
            },
        },
    }

    if source_path.suffixes[-2:] == [".pkl", ".gz"]:
        bioassembly_path = source_path
    else:
        cache_dir = cache_root / _safe_stem(str(candidate.get("candidate_id", source_path.stem)))
        expected = cache_dir / f"{source_path.stem}.pkl.gz"
        if expected.exists():
            input_dict["condition"]["structure_file"] = str(expected)
            bioassembly_path = expected
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            convert_to_bioassembly_dict(input_dict, out_dir=str(cache_dir))
            bioassembly_path = _bioassembly_path_from_cache(input_dict, cache_dir)

    chain_ids = _chain_ids_from_bioassembly_pkl(bioassembly_path)
    _, resolved_binder_chain = _resolve_design_chains(
        chain_ids=chain_ids,
        target_chain=target_chain,
        binder_chain=binder_chain,
    )
    input_dict["condition"]["structure_file"] = str(bioassembly_path)
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
    binder_mask = torch.tensor(
        (atom_array.chain_id == resolved_binder_chain).astype(int), dtype=torch.float32
    )
    if float(binder_mask.sum()) <= 0.0:
        raise ValueError(
            f"Binder mask is empty for {source_path}; "
            f"binder_chain={resolved_binder_chain}, processed_chains={processed_chain_ids}"
        )

    coord = torch.tensor(atom_array.coord, dtype=torch.float32)
    atom_mask = torch.tensor(atom_array.is_resolved.astype(int), dtype=torch.float32)
    return data["input_feature_dict"], coord, atom_mask, binder_mask, _atom_signature(atom_array)


class PreferencePairDataset(Dataset):
    """Dataset for offline-preprocessed diffusion preference pairs."""

    def __init__(self, pairs_jsonl: str) -> None:
        self.pairs_jsonl = Path(pairs_jsonl)
        if not self.pairs_jsonl.exists():
            raise FileNotFoundError(f"Pair file not found: {self.pairs_jsonl}")

        self.records: list[dict[str, Any]] = []
        with self.pairs_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"No valid lines found in {self.pairs_jsonl}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        target_key = rec.get("target_key")
        if target_key is None:
            target_key = rec.get("meta", {}).get("target_key", "unknown_target")
        item: dict[str, Any] = {
            "input_feature_dict": _load_feature_dict(rec["input_feature_path"]),
            "chosen_coord": _load_tensor(rec["chosen_coord_path"]).float(),
            "rejected_coord": _load_tensor(rec["rejected_coord_path"]).float(),
            "pair_id": rec.get("pair_id", str(idx)),
            "target_key": target_key,
        }

        if "atom_mask_path" in rec:
            item["atom_mask"] = _load_tensor(rec["atom_mask_path"]).float()
        else:
            # Default: all atoms valid.
            item["atom_mask"] = torch.ones(item["chosen_coord"].shape[0], dtype=torch.float32)

        if "binder_mask_path" in rec:
            item["binder_mask"] = _load_tensor(rec["binder_mask_path"]).float()
        else:
            # Optional, only used when dpo.use_binder_weight=true.
            item["binder_mask"] = torch.zeros(item["chosen_coord"].shape[0], dtype=torch.float32)
        return item


class PreferenceTargetGroupDataset(Dataset):
    """Dataset for offline-preprocessed target-level candidate groups."""

    def __init__(
        self,
        groups_jsonl: str,
        max_candidates_per_group: int | None = None,
        min_candidates_per_group: int = 2,
        max_atoms_per_group: int | None = None,
    ) -> None:
        self.groups_jsonl = Path(groups_jsonl)
        self.min_candidates_per_group = max(2, int(min_candidates_per_group))
        self.max_atoms_per_group = max_atoms_per_group
        self.runtime_cache_root = self.groups_jsonl.parent / "runtime_bioassembly"
        if not self.groups_jsonl.exists():
            raise FileNotFoundError(f"Group file not found: {self.groups_jsonl}")

        self.records: list[dict[str, Any]] = []
        with self.groups_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                candidates = rec.get("candidates", [])
                if not isinstance(candidates, list):
                    raise TypeError(
                        f"'candidates' must be a list in {self.groups_jsonl}: {type(candidates)}"
                    )
                if max_candidates_per_group is not None:
                    candidates = candidates[: max(0, max_candidates_per_group)]
                if len(candidates) < self.min_candidates_per_group:
                    continue
                if self.max_atoms_per_group is not None:
                    first_path = (
                        candidates[0].get("structure_path")
                        or candidates[0].get("bioassembly_path")
                        or candidates[0].get("coord_path")
                    )
                    if first_path:
                        atom_count = _count_structure_atoms(first_path)
                        if atom_count is not None and atom_count > self.max_atoms_per_group:
                            continue
                rec = dict(rec)
                rec["candidates"] = candidates
                self.records.append(rec)

        if not self.records:
            raise ValueError(f"No valid target groups found in {self.groups_jsonl}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        target_key = rec.get("target_key")
        if target_key is None:
            target_key = rec.get("meta", {}).get("target_key", "unknown_target")

        candidates = rec["candidates"]
        if candidates and "coord_path" in candidates[0]:
            coord_list = [_load_tensor(x["coord_path"]).float() for x in candidates]
            candidate_coord = torch.stack(coord_list, dim=0)
            candidate_score = torch.tensor(
                [float(x.get("score", 0.0)) for x in candidates], dtype=torch.float32
            )
            candidate_rank = torch.tensor(
                [int(x.get("rank", i)) for i, x in enumerate(candidates)], dtype=torch.long
            )
            candidate_valid_mask = torch.ones(len(candidates), dtype=torch.float32)
            candidate_ids = [str(x.get("candidate_id", i)) for i, x in enumerate(candidates)]
        else:
            meta = rec.get("meta", {})
            use_msa = bool(meta.get("use_msa", False))
            target_chain = meta.get("target_chain")
            binder_chain = meta.get("binder_chain")
            cache_root = Path(meta.get("runtime_cache_root", self.runtime_cache_root))

            anchor_signature: str | None = None
            input_feature_dict = None
            atom_mask = None
            binder_mask = None
            coord_list = []
            score_list = []
            rank_list = []
            candidate_ids = []
            failures = 0

            for i, candidate in enumerate(candidates):
                try:
                    feat, coord, cur_atom_mask, cur_binder_mask, atom_signature = (
                        _build_candidate_features_and_labels(
                            candidate,
                            cache_root=cache_root / _safe_stem(str(target_key)),
                            target_chain=target_chain,
                            binder_chain=binder_chain,
                            use_msa=use_msa,
                        )
                    )
                except Exception:
                    failures += 1
                    continue

                if anchor_signature is None:
                    anchor_signature = atom_signature
                    input_feature_dict = feat
                    atom_mask = cur_atom_mask
                    binder_mask = cur_binder_mask
                if atom_signature != anchor_signature:
                    failures += 1
                    continue

                coord_list.append(coord)
                score_list.append(float(candidate.get("score", 0.0)))
                rank_list.append(int(candidate.get("rank", i)))
                candidate_ids.append(str(candidate.get("candidate_id", i)))

            if input_feature_dict is None or atom_mask is None or binder_mask is None:
                raise RuntimeError(f"No valid candidates could be featurized for group {target_key}.")

            candidate_coord = torch.stack(coord_list, dim=0)
            candidate_score = torch.tensor(score_list, dtype=torch.float32)
            candidate_rank = torch.tensor(rank_list, dtype=torch.long)
            candidate_valid_mask = torch.ones(len(coord_list), dtype=torch.float32)
            if len(coord_list) < self.min_candidates_per_group:
                candidate_valid_mask.zero_()

        if "input_feature_path" in rec:
            input_feature_dict = _load_feature_dict(rec["input_feature_path"])
        elif "bioassembly_path" in rec:
            input_feature_dict = _build_feature_dict_from_bioassembly(
                rec["bioassembly_path"],
                use_msa=bool(rec.get("meta", {}).get("use_msa", False)),
            )
        elif "input_feature_dict" not in locals():
            raise KeyError("Target group record must contain input_feature_path or bioassembly_path.")

        item: dict[str, Any] = {
            "input_feature_dict": input_feature_dict,
            "candidate_coord": candidate_coord,
            "candidate_score": candidate_score,
            "candidate_rank": candidate_rank,
            "candidate_valid_mask": candidate_valid_mask,
            "candidate_id": candidate_ids,
            "group_id": rec.get("group_id", rec.get("target_key", str(idx))),
            "target_key": target_key,
        }

        if "atom_mask_path" in rec:
            item["atom_mask"] = _load_tensor(rec["atom_mask_path"]).float()
        else:
            item["atom_mask"] = torch.ones(candidate_coord.shape[1], dtype=torch.float32)

        if "binder_mask_path" in rec:
            item["binder_mask"] = _load_tensor(rec["binder_mask_path"]).float()
        else:
            item["binder_mask"] = torch.zeros(candidate_coord.shape[1], dtype=torch.float32)
        return item


def _stack_feature_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = items[0].keys()
    for k in keys:
        vals = [x[k] for x in items]
        v0 = vals[0]
        if torch.is_tensor(v0):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(v0, dict):
            out[k] = _stack_feature_dict(vals)  # type: ignore[arg-type]
        else:
            out[k] = vals
    return out


def pair_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_feature_dict": _stack_feature_dict([x["input_feature_dict"] for x in batch]),
        "chosen_coord": torch.stack([x["chosen_coord"] for x in batch], dim=0),
        "rejected_coord": torch.stack([x["rejected_coord"] for x in batch], dim=0),
        "atom_mask": torch.stack([x["atom_mask"] for x in batch], dim=0),
        "binder_mask": torch.stack([x["binder_mask"] for x in batch], dim=0),
        "pair_id": [x["pair_id"] for x in batch],
        "target_key": [x["target_key"] for x in batch],
    }


def _pad_stacked_tensor_list(
    tensors: list[torch.Tensor], pad_value: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("Expected at least one tensor to pad.")
    max_len = max(int(t.shape[0]) for t in tensors)
    out_shape = (len(tensors), max_len, *tensors[0].shape[1:])
    out = tensors[0].new_full(out_shape, pad_value)
    valid = torch.zeros(len(tensors), max_len, dtype=torch.float32)
    for i, tensor in enumerate(tensors):
        cur_len = int(tensor.shape[0])
        out[i, :cur_len] = tensor
        valid[i, :cur_len] = 1.0
    return out, valid


def target_group_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_coord, candidate_presence = _pad_stacked_tensor_list(
        [x["candidate_coord"] for x in batch], pad_value=0.0
    )
    candidate_score, _ = _pad_stacked_tensor_list(
        [x["candidate_score"] for x in batch], pad_value=float("-inf")
    )
    candidate_rank, _ = _pad_stacked_tensor_list(
        [x["candidate_rank"] for x in batch], pad_value=float(10**9)
    )
    candidate_valid_mask, _ = _pad_stacked_tensor_list(
        [x["candidate_valid_mask"] for x in batch], pad_value=0.0
    )
    candidate_valid_mask = candidate_valid_mask * candidate_presence

    max_candidates = candidate_coord.shape[1]
    padded_candidate_ids: list[list[str]] = []
    for x in batch:
        cur = list(x["candidate_id"])
        cur += [""] * (max_candidates - len(cur))
        padded_candidate_ids.append(cur)

    return {
        "input_feature_dict": _stack_feature_dict([x["input_feature_dict"] for x in batch]),
        "candidate_coord": candidate_coord,
        "candidate_score": candidate_score,
        "candidate_rank": candidate_rank.long(),
        "candidate_valid_mask": candidate_valid_mask,
        "atom_mask": torch.stack([x["atom_mask"] for x in batch], dim=0),
        "binder_mask": torch.stack([x["binder_mask"] for x in batch], dim=0),
        "candidate_id": padded_candidate_ids,
        "group_id": [x["group_id"] for x in batch],
        "target_key": [x["target_key"] for x in batch],
    }


class TargetBatchSampler(BatchSampler):
    """
    Sample batches where all examples come from the same target.
    """

    def __init__(
        self,
        target_to_indices: dict[str, list[int]],
        batch_size: int,
        drop_last: bool,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.target_to_indices = target_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self._targets = sorted(self.target_to_indices.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        targets = self._targets[:]
        rng.shuffle(targets)
        batches: list[list[int]] = []
        for t in targets:
            idxs = self.target_to_indices[t][:]
            rng.shuffle(idxs)
            if self.drop_last:
                n = (len(idxs) // self.batch_size) * self.batch_size
                idxs = idxs[:n]
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.world_size > 1 and batches:
            remainder = len(batches) % self.world_size
            if remainder:
                pad = self.world_size - remainder
                batches.extend(batches[:pad])

        for batch_index, batch in enumerate(batches):
            if batch_index % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        total = 0
        for idxs in self.target_to_indices.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += (len(idxs) + self.batch_size - 1) // self.batch_size
        if self.world_size > 1 and total:
            remainder = total % self.world_size
            if remainder:
                total += self.world_size - remainder
        return total // self.world_size


def build_dataloader(
    data_format: str,
    pairs_jsonl: str | None,
    groups_jsonl: str | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    group_by_target: bool = True,
    seed: int = 42,
    max_candidates_per_group: int | None = None,
    min_candidates_per_group: int = 2,
    max_atoms_per_group: int | None = None,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> DataLoader:
    if data_format == "pair":
        if not pairs_jsonl:
            raise ValueError("pairs_jsonl is required when data_format='pair'.")
        ds: Dataset = PreferencePairDataset(pairs_jsonl=pairs_jsonl)
        collate_fn = pair_collate_fn
    elif data_format == "target_group":
        if not groups_jsonl:
            raise ValueError("groups_jsonl is required when data_format='target_group'.")
        ds = PreferenceTargetGroupDataset(
            groups_jsonl=groups_jsonl,
            max_candidates_per_group=max_candidates_per_group,
            min_candidates_per_group=min_candidates_per_group,
            max_atoms_per_group=max_atoms_per_group,
        )
        collate_fn = target_group_collate_fn
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")

    if group_by_target:
        target_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(ds.records):  # type: ignore[attr-defined]
            t = rec.get("target_key", rec.get("meta", {}).get("target_key", "unknown_target"))
            target_to_indices[str(t)].append(i)
        batch_sampler = TargetBatchSampler(
            target_to_indices=target_to_indices,
            batch_size=batch_size,
            drop_last=False,
            seed=seed,
            rank=distributed_rank,
            world_size=distributed_world_size,
        )
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    sampler = None
    shuffle = True
    if distributed_world_size > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
