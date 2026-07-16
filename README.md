# PBD-RL

PBD-RL contains GRPO fine-tuning and inference utilities for PXDesign-based protein binder backbone design.

Large artifacts are not stored in this repository:

- PXDesign base checkpoints
- GRPO fine-tuned checkpoints
- training datasets
- generated PXDesign / IF / Protenix intermediate files

Download these artifacts separately and set their local paths in the YAML configs.

## Inference

PXDesign already supports custom checkpoint loading through its underlying inference config. The Click help for `pxdesign infer` does not show every forwarded argument, but the following works:

```bash
pxdesign infer \
  -i /path/to/design.yaml \
  -o /path/to/output \
  --dtype bf16 \
  --N_sample 1 \
  --N_step 400 \
  --eta_type const --eta_min 2.5 --eta_max 2.5 \
  --model_name final \
  --load_checkpoint_dir /path/to/checkpoint_dir \
  --num_workers 0
```

The checkpoint file must be:

```text
/path/to/checkpoint_dir/final.pt
```

Use the config wrapper for reproducible runs:

```bash
cp configs/inference.yaml configs/inference.local.yaml
# edit configs/inference.local.yaml
bash scripts/run_pxdesign_infer.sh configs/inference.local.yaml
```

Check command construction without launching inference:

```bash
bash scripts/run_pxdesign_infer.sh configs/inference.yaml --dry-run
```

The wrapper supports either one input file or a directory of PXDesign YAML/JSON inputs.

## External Artifact Layout

Recommended external release layout:

```text
checkpoints/
  pxdesign_v0.1.0.pt
  grpo_iptm_only/final.pt
  grpo_ptm_only/final.pt
  grpo_balanced/final.pt
datasets/
  binder_grpo_dataset/
examples/
  reppi226_yaml/
  pdb2026_yaml/
```

Then point `configs/inference.local.yaml` to the downloaded paths.

## Notes

- `--num_workers 0` is the safest default for long inference sweeps on the current server setup.
- `pxdesign pipeline` also forwards unknown arguments, so `--model_name` and `--load_checkpoint_dir` can be appended there as well. The first public entrypoint here focuses on raw backbone inference via `pxdesign infer`.
- Generated CIFs generally contain target chain `A0` and binder chain `B0`; downstream PDB conversion workflows have used binder chain `B`.
