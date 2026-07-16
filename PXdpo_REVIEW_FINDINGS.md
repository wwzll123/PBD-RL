# PBD-RL Code and Data Review Findings

Review date: 2026-05-18  
Code path: `/root/PXDesign/PBD-RL`  
Data path: `/root/autodl-tmp/binder_grpo_dataset`

## Scope

This review followed the guidance in:

- `GRPO_CODE_REPORT.md`
- `PROJECT_OUTLINE_AND_REVIEW_PROMPT.md`

The review focused on correctness risks in the offline diffusion-GRPO path:

- data and feature construction
- PXDesign vs Protenix structure source separation
- chain and mask handling
- dataset and collate shapes
- GRPO loss math and masking
- DDP, AMP, skip, and checkpoint behavior
- flat dataset integrity

No files were modified during the review itself.

## Findings

### P0

No confirmed P0 issue was found in the default documented path.

The default group builder uses `complex_pdb_path`, and the inspected flat TSV has all `complex_pdb_path` values under:

```text
/root/autodl-tmp/binder_grpo_dataset/PXDesign_Gen
```

No default-path evidence was found that `pred_pdb_path` or `Protenix_Pred` is used as the training feature or coordinate source.

### P1

#### 1. DDP exception skip can desynchronize ranks

Location: `train.py`, main training loop exception handling.

Problem:

The no-signal skip path synchronizes ranks with an `all_reduce` skip flag, but exceptions raised during forward/loss computation are caught locally and followed by `continue`. In DDP, one rank may skip while another rank enters backward, which can hang on distributed collectives.

Why it matters:

This is a correctness and operability risk for multi-GPU training. It can turn data/model errors on one rank into a silent distributed hang.

Suggested fix:

- Distinguish OOM from unexpected correctness errors.
- In DDP, synchronize an error flag before deciding whether any rank proceeds to backward.
- Prefer fail-fast for non-OOM exceptions, especially while validating the new GRPO path.

#### 2. Default training config points to missing group artifacts

Location: `configs/config.yaml`, `data.groups_jsonl`.

Problem:

The default config points to:

```text
/root/autodl-tmp/tmp/grpo_groups/train_groups.jsonl
```

At review time, that file and `build_groups_stats.json` did not exist.

Why it matters:

Running the default training command before building groups will fail. This is especially confusing because the canonical dataset entry is the flat TSV, while training consumes generated group/cache artifacts.

Suggested fix:

- Add a startup preflight check with a clear error explaining the required group-building command.
- Consider documenting the two-stage flow directly in `configs/config.yaml`.
- Optionally add a config field for the source TSV and generated group directory to make provenance explicit.

#### 3. `structure_path_column` can be misconfigured to use Protenix predictions as features

Location: `make_prefer_pair/build_groups_from_metrics_tsv.py`, `_read_ranked_rows()` and `build_groups()`.

Problem:

The default `--structure_path_column=complex_pdb_path` is correct. However, the CLI permits any column name. If a user passes `--structure_path_column pred_pdb_path`, the feature and coordinate cache will be built from `Protenix_Pred`, violating the project contract.

Why it matters:

The data contract says:

- `PXDesign_Gen` is the training feature and coordinate source.
- `Protenix_Pred` is only reward/provenance.

This invariant is currently convention-based, not enforced.

Suggested fix:

- Hard fail unless `structure_path_column == "complex_pdb_path"`, or
- Validate that every resolved structure path is under `PXDesign_Gen` and not under `Protenix_Pred`.

### P2

#### 4. `batch_size > 1` is unsafe for variable atom counts

Location: `data.py`, `pair_collate_fn()` and `target_group_collate_fn()`.

Problem:

The collate functions pad candidate count but do not pad atom count. `atom_mask`, `binder_mask`, and coordinates are stacked directly. Different targets usually have different `N_atom`, so `batch_size > 1` can fail with a shape error.

Why it matters:

The default `batch_size=1` avoids this, but increasing batch size is a natural tuning step. The code does not clearly enforce the limitation.

Suggested fix:

- Add an explicit assertion that target-group GRPO currently requires `batch_size=1`, or
- Implement atom-dimension padding and verify all downstream model feature shapes support it.

#### 5. `global_step` does not represent optimizer updates

Location: `train.py`, main training loop.

Problem:

`global_step` is incremented for each dataloader batch before skip checks. Skipped batches, exceptions, and gradient accumulation all make `global_step` diverge from actual optimizer updates.

Why it matters:

Checkpoint names, logs, and final checkpoint metadata can be misleading. With `grad_accum_steps > 1`, skip behavior may also make accumulation boundaries hard to reason about.

Suggested fix:

- Track separate counters such as `data_step`, `micro_step`, and `optim_step`.
- Use `optim_step` for checkpoint naming and training-progress reporting.

#### 6. `reference_free=true` still constructs a full reference model

Location: `train.py`, reference model initialization.

Problem:

The code always runs:

```python
reference = copy.deepcopy(policy).to(device)
```

even when `dpo.reference_free=true`.

Why it matters:

Reference-free training still pays the memory cost of a full frozen model copy.

Suggested fix:

- Only construct `reference` when `reference_free` is false.
- Use `None` as the reference placeholder in reference-free mode.

#### 7. `atom_signature` and shared anchor feature have an implicit sequence assumption

Location: `make_prefer_pair/build_groups_from_metrics_tsv.py`, `_atom_signature()` and group writing.

Problem:

The builder groups candidates by atom layout and stores one anchor `input_feature_dict` for the whole group. The atom signature includes chain id, residue id, and atom name, but not residue name or TSV `binder_sequence`.

Data observation:

In sampled `PXDesign_Gen` structures, binder chain `B` is poly-Gly, while TSV `binder_sequence` contains the inverse-folded binder sequence. Lengths match in sampled examples, but residue identities do not.

Why it matters:

If the diffusion training path is intended to condition on actual binder sequence, using the anchor feature from a poly-Gly backbone would be semantically wrong. If the intended task is backbone-only diffusion conditioned on target and geometry, this may be acceptable, but it should be explicit.

Suggested fix:

- Document whether binder sequence is intentionally ignored by GRPO training.
- If sequence should matter, include sequence/residue identity in feature construction and in the group consistency signature.
- Add a validation check comparing TSV sequence length to binder chain residue count.

#### 8. Padded candidates still run through the diffusion module

Location: `modeling.py`, `diffusion_recon_loss_multi_candidates()`.

Problem:

All candidates up to the padded `C` dimension are expanded and passed through diffusion. Invalid candidates are masked only after the per-sample loss is computed.

Why it matters:

This is numerically masked but wastes memory and compute, especially for uneven group sizes.

Suggested fix:

- Keep as-is for simplicity if memory is acceptable.
- For larger runs, consider packed valid candidates or group-size bucketing.

### P3

#### 9. Missing binder masks default to all-zero masks

Location: `data.py`, `PreferencePairDataset` and `PreferenceTargetGroupDataset`.

Problem:

If `binder_mask_path` is absent, the dataset silently uses an all-zero binder mask.

Why it matters:

With `dpo.use_binder_weight=true`, this effectively treats all atoms as non-binder atoms and changes loss weighting.

Suggested fix:

- Warn or hard fail when binder weighting is enabled but no binder mask exists.

#### 10. Some comments and descriptions can be more explicit about feature provenance

Location: `make_prefer_pair/build_groups_from_metrics_tsv.py`, CLI description and metadata fields.

Problem:

Some wording emphasizes Protenix metrics but does not strongly state that structure features must come from PXDesign generated complexes.

Why it matters:

The distinction between reward provenance and training feature source is central to this project.

Suggested fix:

- Update CLI/help text and group metadata to explicitly say: feature source must be `complex_pdb_path` from `PXDesign_Gen`.

## Data Review Summary

Inspected dataset:

```text
/root/autodl-tmp/binder_grpo_dataset/metadata/integrated_metrics.flat_paths.tsv
```

Observed statistics:

- TSV size: about 170 MB.
- Total rows: 382,268.
- Targets: 29,352.
- `complex_pdb_path` missing or nonexistent: 0.
- `pred_pdb_path` missing or nonexistent: 0.
- Empty `binder_sequence`: 0.
- Duplicate `(target, sequence_id)`: 0.
- `chainB_pdb_path` non-empty rows: 0.
- `PXDesign_Gen` PDB count: 382,268.
- `Protenix_Pred` PDB count: 382,268.

Target candidate counts:

- Minimum: 1.
- Median: 10.
- Maximum: 30.
- Targets with fewer than 4 candidates: 193.
- Targets with zero or tiny reward standard deviation: 6, all single-candidate targets in the inspected scoring formula.

Reward metric ranges:

- `ptx_iptm`: min 0.0216, median 0.1707, max 0.9736.
- `ptx_ptm_binder`: min 0.0982, median 0.8452, max 0.9801.
- `ptx_plddt`: min 30.691, median 78.6876, max 96.7914.
- `ptx_pred_design_rmsd`: min 0.73, median 28.47, max 272.64.

Path provenance:

- All `complex_pdb_path` values point to `PXDesign_Gen`.
- All `pred_pdb_path` values point to `Protenix_Pred`.

Sampled PDB chain observations:

- `PXDesign_Gen` examples use chains `A` and `B`.
- Chain `A` corresponds to target.
- Chain `B` corresponds to generated binder backbone.
- `Protenix_Pred` examples also use chains `A` and `B`, but atom counts differ from `PXDesign_Gen`, as expected for predicted all-atom structures.

## Open Questions

1. Should GRPO diffusion training condition on the actual `binder_sequence`, or is it intentionally backbone-only with poly-Gly design structures?
2. Is `batch_size=1` a hard design invariant for target-group GRPO, or should atom-dimension padding be supported?
3. Should non-OOM exceptions during training be skipped, or should they fail fast during this stage of development?
4. Should `pred_pdb_path` ever be allowed as a structure source for any auxiliary experiment, or should it be forbidden in this builder?

## Recommended Tests

1. Add a builder test that rejects `--structure_path_column pred_pdb_path`.
2. Add a chain-mask test using A/B and A0/B0 examples, asserting non-empty binder masks.
3. Add a group-building smoke test on a small flat TSV subset and verify all cached feature paths come from `PXDesign_Gen`.
4. Add a collate test showing that mixed `N_atom` groups fail with a clear message or are padded correctly.
5. Add a GRPO loss test covering padding, `-inf` padded scores, single-valid candidate skip, and zero reward variance.
6. Add a 2-rank DDP smoke test where one rank raises an exception or sees no-signal data, verifying no hang.
7. Add a reference-free memory/path test asserting no reference model is constructed when `dpo.reference_free=true`.

## Summary

The main offline GRPO path is directionally coherent, and the flat dataset appears complete and consistent with the documented path conventions. The highest-priority fixes are DDP-safe exception handling, preflight validation for missing generated group artifacts, and hard enforcement that training features come from `PXDesign_Gen` rather than `Protenix_Pred`.

