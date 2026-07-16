# PBD-RL Diffusion-GRPO 代码报告

生成日期：2026-05-18  
项目路径：`/root/PXDesign/PBD-RL`  
当前推荐数据入口：`/root/autodl-tmp/binder_grpo_dataset/metadata/integrated_metrics.flat_paths.tsv`

## 1. 项目目标

当前 `PBD-RL` 已从原始的离线 diffusion-DPO 训练代码扩展为支持离线 diffusion-GRPO 的训练框架。项目的核心任务是：

1. 针对每个 target，PXDesign 生成多个 binder backbone。
2. 对 binder 进行反向折叠得到序列。
3. 使用 Protenix/AF3 类流程折叠 target-binder 二聚体复合物。
4. 从折叠结果中提取 reward 指标，例如 `ptx_iptm`、`ptx_ptm_binder`、`ptx_plddt`。
5. 对同一个 target 下的一组候选 binder 做 group-relative policy optimization。

训练中使用 diffusion reconstruction loss 的负值作为 `logp` surrogate，即：

```text
logp_surrogate = - diffusion_reconstruction_loss
```

GRPO 部分不再构造成 chosen/rejected pair，而是在每个 target group 内根据 reward 做标准化 advantage。

## 2. 当前数据集约定

当前整理后的干净数据集只依赖三类文件：

```text
/root/autodl-tmp/binder_grpo_dataset/
  PXDesign_Gen/
  Protenix_Pred/
  metadata/
    integrated_metrics.flat_paths.tsv
```

### 2.1 `PXDesign_Gen`

`PXDesign_Gen` 保存 PXDesign 生成的 target+binder 设计复合物结构，是训练特征和坐标标签的来源。

命名规则：

```text
TargetName_id_N.pdb
```

例如：

```text
/root/autodl-tmp/binder_grpo_dataset/PXDesign_Gen/11as_A_vs_B_id_10.pdb
```

重要链名约定：

```text
A / A0 = target all-atom coordinate
B / B0 = generated binder backbone
```

`target_D_vs_C` 这类 target 名只表示原始 PDB 解析语义：D 链是 target，C 链用于定义接触/hotspot。进入 PXDesign 输出后，训练结构统一按 `A/A0` 和 `B/B0` 解释，不再从 target 名解析训练链名。

### 2.2 `Protenix_Pred`

`Protenix_Pred` 保存 Protenix 预测出的 target-binder 二聚体复合物结构，用于 reward/provenance 追溯。

命名规则同样是：

```text
TargetName_id_N.pdb
```

例如：

```text
/root/autodl-tmp/binder_grpo_dataset/Protenix_Pred/11as_A_vs_B_id_10.pdb
```

### 2.3 `integrated_metrics.flat_paths.tsv`

这是当前训练数据构建的唯一推荐 TSV 入口。

关键列：

| 列名 | 功能 |
| --- | --- |
| `target` | target group 名，用于按 target 聚合候选 |
| `sequence_id` | candidate id，例如 `id_10_design_0` |
| `complex_pdb_path` | 指向 `PXDesign_Gen`，训练 feature/coord 来源 |
| `pred_pdb_path` | 指向 `Protenix_Pred`，reward/provenance 结构 |
| `binder_sequence` | binder 序列，已直接保存在 TSV |
| `ptx_iptm` | reward 主指标 |
| `ptx_ptm_binder` | binder 侧结构质量指标 |
| `ptx_plddt` | 置信度指标 |
| `ptx_pred_design_rmsd` | 当前默认不参与 reward |

`chainB_pdb_path` 在 flat TSV 中已置空。训练和当前 reward 构建不依赖单链 B 文件。

## 3. 数据构建模块

文件：

```text
make_prefer_pair/build_groups_from_metrics_tsv.py
```

该脚本将 flat TSV 转成训练用的 `train_groups.jsonl` 和对应的 tensor cache。

### 3.1 输入

推荐命令：

```bash
python make_prefer_pair/build_groups_from_metrics_tsv.py \
  --metrics_tsv /root/autodl-tmp/binder_grpo_dataset/metadata/integrated_metrics.flat_paths.tsv \
  --out_dir /root/autodl-tmp/tmp/grpo_groups \
  --max_candidates_per_target 24 \
  --min_candidates_per_group 4 \
  --selection_strategy stratified
```

### 3.2 reward 计算

当前默认 reward：

```text
score = 1.0 * ptx_iptm
      + 0.2 * ptx_ptm_binder
      + 0.2 * (ptx_plddt / 100)
      - 0.0 * log1p(ptx_pred_design_rmsd)
```

也就是说，RMSD 默认权重为 `0.0`，当前实际不参与训练 reward。

原因：`ptx_pred_design_rmsd` 是 complex2complex RMSD，容易被复合物整体对齐和 target 差异主导，不是 binder2binder RMSD。若后续用 USalign 重新计算 binder2binder RMSD，可以再以小权重加入。

### 3.3 特征化流程

对 TSV 中每个 candidate：

1. 读取 `complex_pdb_path` 指向的 PXDesign 设计复合物。
2. 调用 PXDesign 的 `convert_to_bioassembly_dict` 转换成 bioassembly cache。
3. 用 `InferenceDataset.process_sample_dict` 和 `process_one` 生成 `input_feature_dict`。
4. 保存：
   - `input_feature.pt`
   - `coord.pt`
   - `atom_mask.pt`
   - `binder_mask.pt`
5. 按 atom signature 分组，确保同一个 group 中候选结构的 atom 排布一致。

链名解析逻辑：

```text
默认只接受 PXDesign 输出约定：
A/B 或 A0/B0
```

如果处理后链名从 `A/B` 变成 `A0/B0`，代码会自动映射。若 binder mask 为空，会抛出异常并跳过该 candidate，避免把错误 mask 写入训练集。

### 3.4 输出

输出目录结构：

```text
grpo_groups/
  train_groups.jsonl
  build_groups_stats.json
  cache_bioassembly/
    Target__Candidate/
  tensors/
    features/
    coords/
    masks/
```

`cache_bioassembly` 采用每个 candidate 一个子目录，避免 `id_N.cif` 跨 target 命名冲突。

## 4. 数据加载模块

文件：

```text
data.py
```

### 4.1 `PreferenceTargetGroupDataset`

这是当前 GRPO 路径使用的 Dataset。

每条样本对应一个 target group：

```text
{
  input_feature_dict,
  candidate_coord: [C, N_atom, 3],
  candidate_score: [C],
  candidate_rank: [C],
  candidate_valid_mask: [C],
  atom_mask: [N_atom],
  binder_mask: [N_atom],
  candidate_id,
  group_id,
  target_key
}
```

其中：

| 字段 | 说明 |
| --- | --- |
| `candidate_coord` | 同一 target 下多个 binder 设计结构坐标 |
| `candidate_score` | TSV reward 计算得到的候选分数 |
| `candidate_rank` | 按 reward 排序的 rank |
| `candidate_valid_mask` | padding 后标记有效候选 |
| `atom_mask` | resolved atom mask |
| `binder_mask` | binder atom 位置，来自 `B/B0` 链 |

### 4.2 `target_group_collate_fn`

不同 group 的 candidate 数可能不同，因此 collate 时会 padding 到 batch 内最大 candidate 数。

输出形状：

```text
candidate_coord       [B, C, N_atom, 3]
candidate_score       [B, C]
candidate_rank        [B, C]
candidate_valid_mask  [B, C]
atom_mask             [B, N_atom]
binder_mask           [B, N_atom]
```

当前默认 `batch_size=1`，即一个 optimization step 处理一个 target group。

### 4.3 `TargetBatchSampler`

`TargetBatchSampler` 保证同一 batch 中样本来自同一 target。DDP 下还会把 batch 数 padding 到 `world_size` 的倍数，避免多卡步数不一致导致 hang。

## 5. 模型封装模块

文件：

```text
modeling.py
```

核心类：

```python
class PXDPOModel(ProtenixDesign)
```

它继承 PXDesign 原始模型，并新增适合 DPO/GRPO 的 diffusion reconstruction loss 接口。

### 5.1 条件编码

```python
encode_condition(input_feature_dict)
```

调用 PXDesign 的：

```python
get_condition_embedding(...)
```

得到：

```text
s_inputs
s_trunk
z_trunk
```

训练中 policy/reference 会分别缓存 condition embedding，避免同一 batch 内重复计算。

### 5.2 噪声并行

当前实现是噪声并行，而不是样本并行。

`diffusion_recon_loss_multi_candidates` 输入：

```text
x0: [B, C, N_atom, 3]
```

其中：

```text
B = batch size
C = candidate 数
K = noise_parallel_steps
```

内部会展开为：

```text
[B, C*K, N_atom, 3]
```

即同一个 candidate 会在多个 diffusion noise level 上并行加噪/去噪，然后对 K 个噪声尺度的 reconstruction loss 取平均。

### 5.3 binder 加权 loss

diffusion MSE 先按 atom 聚合，再按 noise K 聚合。

如果 `dpo.use_binder_weight=true`：

```text
atom_weight = binder_weight * binder_mask
            + non_binder_weight * (1 - binder_mask)
```

默认：

```yaml
binder_weight: 1.0
non_binder_weight: 0.2
```

因此 binder 区域的重构误差权重更高，target 区域仍保留较小权重来稳定复合物上下文。

## 6. Loss 模块

文件：

```text
losses.py
```

### 6.1 legacy DPO

保留了两个 DPO loss：

```python
diffusion_dpo_loss(...)
diffusion_dpo_loss_multi_candidate(...)
```

这些用于 pairwise chosen/rejected 或 top1-vs-rest 风格训练。

### 6.2 `diffusion_grpo_loss`

当前 GRPO 主路径使用：

```python
diffusion_grpo_loss(
    policy_loss,
    ref_loss,
    candidate_score,
    candidate_valid_mask,
    kl_coef,
    advantage_eps,
    advantage_clip,
    reward_clip,
    reference_free,
)
```

输入形状：

```text
policy_loss      [B, C]
ref_loss         [B, C] or None
candidate_score  [B, C]
valid_mask       [B, C]
```

核心计算：

1. 对每个 target group 内的 reward 做均值和标准差：

```text
advantage = (reward - group_mean) / group_std
```

2. 可选裁剪：

```text
advantage_clip = 5.0
```

3. 用 diffusion loss 的负值作为 log-prob surrogate：

```text
policy_logp = -policy_loss
```

4. policy gradient 项：

```text
L_pg = - mean(advantage * policy_logp)
```

5. reference KL-like anchor：

```text
L_kl = mean((policy_logp - ref_logp)^2)
```

6. 总 loss：

```text
L = L_pg + kl_coef * L_kl
```

如果 `dpo.reference_free=true`，则不计算 reference loss，KL 项为 0。

## 7. 训练主流程

文件：

```text
train.py
```

### 7.1 初始化

训练入口使用 Hydra：

```bash
python train.py ...
```

流程：

1. 初始化 DDP。
2. 固定随机种子。
3. 创建输出目录并保存 `resolved_config.yaml`。
4. 构建 PXDesign config。
5. 加载 `PXDPOModel` 和 PXDesign checkpoint。
6. 冻结非训练模块。
7. 深拷贝 policy 得到 reference model。
8. 构建 DataLoader。
9. 进入 epoch/step 训练循环。

### 7.2 默认训练参数

配置文件：

```text
configs/config.yaml
```

关键配置：

```yaml
data:
  data_format: target_group
  max_candidates_per_group: 24
  min_candidates_per_group: 4
  group_by_target: true

model:
  trainable_modules:
    - diffusion_module
    - design_condition_embedder

training:
  distributed: auto
  amp: true
  precision: bf16
  noise_parallel_steps: 8
  num_noise_levels: 400

grpo:
  kl_coef: 0.02
  min_reward_std: 1.0e-8
  advantage_clip: 5.0
```

当前默认不是全量微调，而是只训练：

```text
diffusion_module
design_condition_embedder
```

LoRA/PEFT 目前还没有接入代码路径。

### 7.3 训练一步的核心流程

对 `target_group` 数据：

1. 从 batch 读取：

```text
candidate_coord       [B, C, N_atom, 3]
candidate_score       [B, C]
candidate_valid_mask  [B, C]
atom_mask             [B, N_atom]
binder_mask           [B, N_atom]
```

2. 检查训练信号：

```text
有效 candidate 数 >= 2
reward std > min_reward_std
```

不满足则跳过该 batch。

3. 采样共享噪声：

```text
shared_noise [B, K, N_atom, 3]
shared_sigma [B, K]
```

4. policy 计算所有 candidate 的 diffusion recon loss：

```text
policy_loss [B, C]
```

5. reference model 在 no-grad 下计算：

```text
ref_loss [B, C]
```

若 `reference_free=true`，跳过 reference。

6. 调用 `diffusion_grpo_loss`。

7. 可选加入 SFT 项：

```text
loss = grpo_loss + sft_coef * best_candidate_sft
```

当前默认 `sft_coef=0.0`。

8. AMP/BF16 下 backward、梯度裁剪、optimizer step。

### 7.4 异常和 DDP 防崩机制

训练循环内有多层保护：

| 情况 | 行为 |
| --- | --- |
| group 少于 2 个有效候选 | skip |
| reward 标准差过小 | skip |
| non-finite loss | skip |
| OOM | 清梯度、empty cache、skip |
| DDP 某个 rank 要 skip | all-reduce skip flag，所有 rank 同步 skip |

这样可以减少 DDP debug 中常见的单 rank hang。

## 8. DDP 与 AMP

### 8.1 DDP

配置：

```yaml
training:
  distributed: auto
  ddp_backend: nccl
  ddp_find_unused_parameters: true
```

启动示例：

```bash
torchrun --nproc_per_node=2 train.py \
  data.groups_jsonl=/root/autodl-tmp/tmp/grpo_groups/train_groups.jsonl
```

当 `WORLD_SIZE > 1` 且 `distributed=auto` 时自动启用 DDP。

### 8.2 AMP BF16

配置：

```yaml
training:
  amp: true
  precision: bf16
```

BF16 下不启用 GradScaler。若切换为 FP16，则 GradScaler 会启用。

## 9. 当前推荐运行流程

### 9.1 从 flat TSV 构建 GRPO groups

```bash
cd /root/PXDesign

python make_prefer_pair/build_groups_from_metrics_tsv.py \
  --metrics_tsv /root/autodl-tmp/binder_grpo_dataset/metadata/integrated_metrics.flat_paths.tsv \
  --out_dir /root/autodl-tmp/tmp/grpo_groups \
  --max_candidates_per_target 24 \
  --min_candidates_per_group 4 \
  --selection_strategy stratified
```

### 9.2 单卡 smoke 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  data.groups_jsonl=/root/autodl-tmp/tmp/grpo_groups/train_groups.jsonl \
  data.max_candidates_per_group=2 \
  data.min_candidates_per_group=2 \
  data.num_workers=0 \
  training.max_steps=1 \
  training.noise_parallel_steps=2 \
  training.log_every_steps=1 \
  training.save_every_steps=1000 \
  training.debug_tracebacks=true \
  experiment.output_dir=/root/autodl-tmp/tmp/pxdpo_smoke_outputs \
  experiment.run_name=flat_dataset_single_step \
  dpo.reference_free=true
```

最近一次 flat dataset smoke 已通过：

```text
[epoch=0 step=1] loss=-87.163467 ... reward=0.686533 kl=0.000000 K=2 skipped=0
Training finished. Final checkpoint:
/root/autodl-tmp/tmp/pxdpo_smoke_outputs/flat_dataset_single_step/final.pt
```

## 10. 其他辅助模块

### 10.1 `make_prefer_pair/build_pairs.py`

旧版 DPO pair 构建脚本，生成 chosen/rejected pair。当前 GRPO 主流程不推荐使用，但保留兼容。

### 10.2 `partial_infer.py`

用于 PXDesign 局部/部分扩散采样测试，包含：

- 模型加载
- task dict 特征化
- 固定区域 mask 检查
- partial diffusion sampling
- CIF 输出

该模块偏推理和生成，不属于当前 GRPO 训练主路径。

### 10.3 `run_IF_design.py` 与 `launch_IF_design.py`

用于历史 binder backbone 的 inverse folding 设计流程。当前 flat TSV 已直接保存 `binder_sequence`，训练主路径不依赖 IF 原始目录。

### 10.4 `file_process/*`

包含一些历史数据预处理脚本，例如 YAML、MSA、PKL cache 处理。当前 flat dataset + GRPO 构建主路径不依赖这些脚本。

## 11. 当前限制与建议

1. **LoRA/PEFT 尚未接入。**  
   当前通过 `trainable_modules` 控制部分参数微调。如果后续显存压力大，可以给 `diffusion_module` 或 attention/linear 子模块接 PEFT。

2. **RMSD reward 暂时关闭。**  
   当前 `w_rmsd_log=0.0`。建议后续用 USalign 计算 binder2binder RMSD 或 TM-score 后，再作为 reward 或 filter。

3. **GRPO 是离线 objective。**  
   当前不会在线调用 PXDesign 生成新 binder，也不会在线跑 Protenix 重新评估 reward。

4. **reference model 会占额外显存。**  
   若 `dpo.reference_free=true`，训练时可跳过 reference 计算，但代码仍会构建 reference copy。后续可优化为 reference-free 模式下不创建 reference model。

5. **当前 group 构建会预先保存 tensor cache。**  
   这会占用 `/root/autodl-tmp/tmp` 空间。优点是训练稳定、启动快；缺点是构建阶段需要额外磁盘。

6. **默认 `batch_size=1`。**  
   对蛋白复合物 all-atom diffusion 来说比较稳。扩大 batch size 前需要评估 atom 数差异、显存和 padding 成本。

## 12. 总结

当前代码主链路已经形成：

```text
integrated_metrics.flat_paths.tsv
    -> build_groups_from_metrics_tsv.py
    -> train_groups.jsonl + tensor cache
    -> PreferenceTargetGroupDataset
    -> target_group_collate_fn
    -> PXDPOModel.diffusion_recon_loss_multi_candidates
    -> diffusion_grpo_loss
    -> train.py optimizer step
```

最新实现已经支持：

- flat dataset 数据对接
- PXDesign 输出链名 `A/A0` target、`B/B0` binder
- binder mask 非空校验
- per-candidate bioassembly cache，避免命名冲突
- target-level offline GRPO
- 噪声并行 diffusion training
- DDP
- AMP BF16
- 异常 batch skip 与 DDP skip 同步
- smoke 构建和 1-step 训练验证

