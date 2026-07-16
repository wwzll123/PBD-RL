# PBD-RL Diffusion-GRPO 项目大纲与 Review Prompt

生成日期：2026-05-18  
代码路径：`/root/PXDesign/PBD-RL`  
主报告：`/root/PXDesign/PBD-RL/GRPO_CODE_REPORT.md`

## 1. 项目大纲

### 1.1 目标

本项目将 PXDesign 的 binder diffusion 训练从离线 diffusion-DPO 扩展为离线 diffusion-GRPO。

核心思路：

```text
同一个 target 下有多个 PXDesign 生成的 binder backbone
    -> 使用 Protenix/AF3 类流程评估 target-binder 复合物质量
    -> 从 TSV 中读取 reward 指标
    -> group 内标准化 reward 得到 advantage
    -> 用 diffusion reconstruction loss 的负值作为 logp surrogate
    -> 优化 diffusion-GRPO objective
```

### 1.2 当前数据依赖

当前核心流程只依赖：

```text
/root/autodl-tmp/binder_grpo_dataset/PXDesign_Gen
/root/autodl-tmp/binder_grpo_dataset/Protenix_Pred
/root/autodl-tmp/binder_grpo_dataset/metadata/integrated_metrics.flat_paths.tsv
```

其中：

- `PXDesign_Gen`: PXDesign 生成的 target+binder 设计结构，训练 feature 和 coord 来源。
- `Protenix_Pred`: Protenix 预测的 target-binder 二聚体复合物结构，用于 reward/provenance。
- `integrated_metrics.flat_paths.tsv`: 统一索引表，包含路径、reward 指标和 `binder_sequence`。

PXDesign 输出链名约定：

```text
A / A0 = target
B / B0 = generated binder backbone
```

`target_D_vs_C` 这样的 target 名是原始 PDB/hotspot 语义，不应该用于解释 PXDesign 输出结构中的训练链名。

### 1.3 主流程

```text
integrated_metrics.flat_paths.tsv
    -> make_prefer_pair/build_groups_from_metrics_tsv.py
    -> train_groups.jsonl + tensor cache
    -> data.PreferenceTargetGroupDataset
    -> data.target_group_collate_fn
    -> modeling.PXDPOModel.diffusion_recon_loss_multi_candidates
    -> losses.diffusion_grpo_loss
    -> train.py optimizer step
```

### 1.4 关键模块

#### `make_prefer_pair/build_groups_from_metrics_tsv.py`

职责：

- 流式读取 flat TSV。
- 按 `target` 聚合 candidate。
- 计算 reward。
- 从 `complex_pdb_path` 读取 PXDesign 设计复合物。
- 调用 PXDesign feature pipeline 生成 `input_feature_dict`。
- 保存 `coord`、`atom_mask`、`binder_mask`。
- 按 atom signature 过滤不一致候选。
- 写出 `train_groups.jsonl`。

当前默认 reward：

```text
score = 1.0 * ptx_iptm
      + 0.2 * ptx_ptm_binder
      + 0.2 * (ptx_plddt / 100)
      - 0.0 * log1p(ptx_pred_design_rmsd)
```

RMSD 当前默认关闭。

#### `data.py`

职责：

- 加载 `train_groups.jsonl`。
- 将同一个 target 下多个 candidate 组成 `[C, N_atom, 3]`。
- collate 成 `[B, C, N_atom, 3]`。
- padding candidate 维度并生成 `candidate_valid_mask`。
- DDP 下用 `TargetBatchSampler` 保证每个 rank 步数一致。

#### `modeling.py`

职责：

- 继承 PXDesign `ProtenixDesign`。
- 缓存 condition embedding。
- 对多个 candidate、多噪声尺度并行计算 diffusion reconstruction loss。
- 当前是噪声并行：`[B, C, N_atom, 3] -> [B, C*K, N_atom, 3]`。
- 用 `binder_mask` 对 binder 区域加权。

#### `losses.py`

职责：

- 保留 legacy diffusion-DPO loss。
- 实现 `diffusion_grpo_loss`。

GRPO loss 逻辑：

```text
advantage = (reward - group_mean) / group_std
policy_logp = -policy_diffusion_loss
L_pg = -mean(advantage * policy_logp)
L_kl = mean((policy_logp - ref_logp)^2)
L = L_pg + kl_coef * L_kl
```

#### `train.py`

职责：

- Hydra 配置入口。
- 加载 PXDesign checkpoint。
- 冻结非训练模块。
- 创建 policy/reference model。
- 支持 DDP、AMP BF16。
- 训练 target-level GRPO。
- 对异常 batch 做 skip，避免 DDP hang。

默认可训练模块：

```yaml
model:
  trainable_modules:
    - diffusion_module
    - design_condition_embedder
```

### 1.5 典型命令

构建 groups：

```bash
cd /root/PXDesign

python make_prefer_pair/build_groups_from_metrics_tsv.py \
  --metrics_tsv /root/autodl-tmp/binder_grpo_dataset/metadata/integrated_metrics.flat_paths.tsv \
  --out_dir /root/autodl-tmp/tmp/grpo_groups \
  --max_candidates_per_target 24 \
  --min_candidates_per_group 4 \
  --selection_strategy stratified
```

单卡 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  data.groups_jsonl=/root/autodl-tmp/tmp/grpo_groups/train_groups.jsonl \
  data.max_candidates_per_group=2 \
  data.min_candidates_per_group=2 \
  data.num_workers=0 \
  training.max_steps=1 \
  training.noise_parallel_steps=2 \
  training.log_every_steps=1 \
  training.debug_tracebacks=true \
  dpo.reference_free=true
```

DDP 示例：

```bash
torchrun --nproc_per_node=2 train.py \
  data.groups_jsonl=/root/autodl-tmp/tmp/grpo_groups/train_groups.jsonl
```

## 2. Review Prompt

下面这段可以直接复制给其他代码审查 agent。

```text
你是一个严格的 senior ML systems/code reviewer。请 review `/root/PXDesign/PBD-RL` 这个项目的最新代码，目标是发现 correctness bug、训练逻辑错误、数据处理问题、DDP/AMP 风险、shape/mask 错误和潜在数据泄漏。请优先给出具体文件/函数/行附近的发现，按严重程度排序，不要只做泛泛建议。

项目背景：
- 这是 PXDesign binder diffusion 的离线 diffusion-GRPO 改造。
- 数据集入口是 `/root/autodl-tmp/binder_grpo_dataset/metadata/integrated_metrics.flat_paths.tsv`。
- 当前核心数据只依赖：
  - `/root/autodl-tmp/binder_grpo_dataset/PXDesign_Gen`
  - `/root/autodl-tmp/binder_grpo_dataset/Protenix_Pred`
  - `/root/autodl-tmp/binder_grpo_dataset/metadata/integrated_metrics.flat_paths.tsv`
- `PXDesign_Gen` 中的结构是训练 feature/coord 来源。
- `Protenix_Pred` 中的结构只是 reward/provenance 来源，不应作为训练 feature。
- PXDesign 输出链名约定：`A/A0` 是 target，`B/B0` 是 generated binder。
- `target_D_vs_C` 这类 target 名只表示原始 PDB/hotspot 语义，不应用于解释 PXDesign 输出中的训练链名。

主流程：
1. `make_prefer_pair/build_groups_from_metrics_tsv.py`
   - 读取 flat TSV。
   - 用 `complex_pdb_path` 构建 feature/coord/mask。
   - 按 target 聚合 candidate。
   - 计算 reward。
   - 写出 `train_groups.jsonl` 和 tensor cache。
2. `data.py`
   - `PreferenceTargetGroupDataset` 加载 `train_groups.jsonl`。
   - `target_group_collate_fn` padding candidate 维度。
   - `TargetBatchSampler` 支持 DDP 下 rank 间步数一致。
3. `modeling.py`
   - `PXDPOModel` 继承 PXDesign `ProtenixDesign`。
   - `diffusion_recon_loss_multi_candidates` 对 `[B, C, N_atom, 3]` 做多 candidate、多 noise level reconstruction loss。
   - 当前实现是噪声并行：展开成 `[B, C*K, N_atom, 3]`。
4. `losses.py`
   - `diffusion_grpo_loss` 用 `-diffusion_loss` 作为 logp surrogate。
   - group 内标准化 reward 得到 advantage。
   - loss = policy gradient term + KL-like anchor。
5. `train.py`
   - Hydra 配置入口。
   - 支持 DDP、AMP BF16。
   - 异常 batch skip，DDP skip 同步。

请重点 review 以下风险：

一、数据与特征构建
- `build_groups_from_metrics_tsv.py` 是否一定只用 `complex_pdb_path` 作为 feature source？
- 是否可能误用 `pred_pdb_path` 或 Protenix predicted complex 作为训练 feature？
- `A/A0 -> target`、`B/B0 -> binder` 的链名解析是否稳健？
- `binder_mask` 是否可能全 0、错链、或与 `coord` atom order 不一致？
- `atom_signature` 过滤是否足够保证同一 group 内 candidate atom layout 一致？
- per-candidate cache 是否仍可能有命名冲突或复用错误？
- `selection_strategy=stratified` 是否会导致 reward 分布或训练信号异常？
- `w_rmsd_log=0.0` 是否真正让 RMSD 不影响 reward？
- 如果某个 target 只有一个 candidate 或 reward 全相等，是否会安全跳过？

二、Dataset/DataLoader
- `PreferenceTargetGroupDataset` 的 tensor shape 是否和模型期望一致？
- padding 后的 `candidate_score=-inf` 是否可能在 loss 或 signal check 中产生问题？
- `candidate_valid_mask` 是否正确屏蔽 padding candidate？
- `TargetBatchSampler` 的 DDP padding 是否可能重复 batch 并影响统计或导致 rank 间不一致？
- `batch_size>1` 时，不同 target 的 `N_atom` 是否可能无法 stack？当前代码是否隐含只支持同 atom 数或 batch_size=1？

三、模型与 diffusion loss
- `diffusion_recon_loss_multi_candidates` 中 `[B, C, K]` 展开为 `[B, C*K]` 是否与 PXDesign diffusion module 的 `input_feature_dict` sample 维语义一致？
- `_feature_dict_for_diffusion_samples` 只处理 `atom_to_token_idx` 的 squeeze 是否足够？其他 feature 是否也可能有 batch/sample 维错误？
- `condition_cache` 在 candidate/noise 展开时是否正确广播？
- `fixed_noise` 和 `fixed_sigma` 是否在 policy/reference、candidate 之间共享得合理？
- binder/non-binder weight 是否正确应用，并且 padding/invalid candidate 不会污染 loss？

四、GRPO loss
- `diffusion_grpo_loss` 是否符合离线 GRPO 的合理 surrogate？
- `advantage` 的 group 内标准化是否正确处理 padding、无效值、std 过小？
- `policy_gradient_vec = -(advantage * policy_logp)` 的符号是否正确？
- KL-like anchor `(policy_logp - ref_logp)^2` 是否尺度合理，是否可能过强/过弱？
- `reference_free=true` 时是否还有不必要的 reference model 显存占用？

五、训练流程
- `train.py` 中 global_step 在 skip batch 时是否符合预期？
- skip 逻辑在 DDP 下是否一定不会造成某些 rank backward、某些 rank continue？
- exception skip 是否会掩盖严重 bug？是否应该区分 OOM 和数据错误？
- AMP BF16 autocast 范围是否合理？
- checkpoint 保存的 `policy_core.state_dict()` 在 DDP/non-DDP 下是否一致？
- `trainable_modules` 按 name prefix 匹配是否可能漏训或误训模块？

六、工程与可维护性
- 默认配置是否应该直接指向 flat dataset 或 generated groups？
- 文档命令与实际配置是否一致？
- 临时输出是否都在 `/root/autodl-tmp/tmp` 下，避免写爆系统盘？
- 是否需要为构建脚本、mask 检查、sampler、loss 写单元测试？

输出格式要求：
1. 先列 Findings，按 P0/P1/P2/P3 严重程度排序。
2. 每个 finding 必须包含文件、函数或代码位置、问题描述、为什么会出错、建议修复方式。
3. 然后列 Open Questions。
4. 最后给一个简短 Summary。
5. 如果没有发现严重问题，也要指出 residual risks 和建议补的测试。
```

## 3. 建议 Review 产物

建议让其他 agent 输出：

```text
PBD-RL_REVIEW_FINDINGS.md
```

并至少覆盖：

- 数据构建 correctness
- GRPO 数学符号和 mask 处理
- DDP 同步安全
- batch shape 安全
- 显存/磁盘风险
- flat dataset 路径依赖

