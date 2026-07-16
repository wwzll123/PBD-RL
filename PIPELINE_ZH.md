# PXDPO 数据加载、特征化与训练管道说明

本文档详细说明当前 `PBD-RL` 目录下实现的端到端流程，包括：

- 原始复合物与打分数据如何转换为 DPO 训练样本
- 训练阶段如何加载样本与组 batch
- 如何与 `PXDesign` 的输入输出结构对齐
- `Diffusion-DPO` 的损失与训练细节
- 关键配置项（Hydra）及建议使用方式

---

## 1. 目标与设计原则

当前实现是一个**最小可运行（MVP）** 的 `Diffusion-DPO` 微调管道，目标是：

1. 复用 `PXDesign` 的模型结构（`ProtenixDesign`）与条件特征输入格式
2. 支持离线偏好对（chosen/rejected）训练
3. 支持“每个 step 只训练单一 target”的采样策略
4. 用 `hydra-core` 管理数据路径和训练超参数

代码入口：

- Pair 生成：`make_prefer_pair/build_pairs.py`
- 训练入口：`train.py`
- 数据集：`data.py`
- 模型封装：`modeling.py`
- DPO损失：`losses.py`

---

## 2. 原始输入数据要求

你目前提供的数据形态完全符合本实现预期：

1. `complex` 目录包含所有复合物 CIF，例如：
   - `1a0a_B_0.cif`
   - `1a0a_B_1.cif`
   - ...
2. 分数 JSON（示例）：

```json
{
  "1a0a_B_0": {"pTM": 0.5, "ipTM": 0.6},
  "1a0a_B_1": {"pTM": 0.3, "ipTM": 0.5}
}
```

命名约定要求：

- key 形如 `<target>_<sample_idx>`，例如 `1a0a_B_0`
- 其中 `<target>` 形如 `1a0a_B`（含 target chain 信息）

---

## 3. Pair 生成流程（make_prefer_pair）

脚本：`make_prefer_pair/build_pairs.py`  
配置：`configs/make_pairs.yaml`

### 3.1 流程概览

对每个 target group（如 `1a0a_B`）：

1. 读取该组所有样本的 `pTM/ipTM`
2. 按加权分数排序：  
   `score = w_ptm * pTM + w_iptm * ipTM`
3. 生成偏好对：
   - chosen 来自前半段（top half）
   - rejected 来自后半段（bottom half）
   - 每个 target 最多 `pairs_per_target` 对
4. 对 pair 涉及的每个 CIF 做一次特征化缓存，输出 `.pt`
5. 写入 `train_pairs.jsonl`

### 3.2 CIF -> PXDesign 特征对齐

为了确保与 `PXDesign` 对齐，脚本复用了 `pxdesign` 现有流程：

1. `convert_to_bioassembly_dict(...)`  
   将 CIF 转成 `bioassembly`（`.pkl.gz`）
2. 通过 `InferenceDataset.process_sample_dict(...)` 与 `process_one(...)`  
   得到 `input_feature_dict`（结构与 PXDesign 推理一致）

这一步保证训练输入与 `pxdesign/runner/inference.py` 中模型调用保持兼容。

### 3.3 生成的缓存文件

默认输出目录：`make_pairs.output.out_dir`（例如 `./data/pairs`）

- `train_pairs.jsonl`
- `tensors/features/*_input_feature.pt`
- `tensors/coords/*_coord.pt`
- `tensors/masks/*_atom_mask.pt`
- `tensors/masks/*_binder_mask.pt`
- `cache_bioassembly/*.pkl.gz`

---

## 4. PreferencePairDataset 加载逻辑

实现文件：`data.py`

### 4.1 单条样本格式（jsonl）

每行至少包含：

- `input_feature_path`
- `chosen_coord_path`
- `rejected_coord_path`

可选：

- `atom_mask_path`
- `binder_mask_path`
- `pair_id`
- `target_key`

当前 `build_pairs.py` 会同时写入这些字段（含 `target_key`）。

### 4.2 Dataset 返回结构

`__getitem__` 返回：

- `input_feature_dict`: `dict[str, Tensor]`
- `chosen_coord`: `[N_atom, 3]`
- `rejected_coord`: `[N_atom, 3]`
- `atom_mask`: `[N_atom]`
- `binder_mask`: `[N_atom]`
- `pair_id`: `str`
- `target_key`: `str`

### 4.3 按 target 分组 batch（核心）

新增了 `TargetBatchSampler`，用于满足你的要求：

- 一个 batch 内样本**全部来自同一个 target**
- 因此一个 step 只训练单一 target 的 chosen/rejected 对

控制开关：

- `data.group_by_target: true`（默认开启）

如果你的 binder 长度固定且原子顺序一致，可以安全地 `batch_size > 1`。

---

## 5. 模型封装与前向对齐

实现文件：`modeling.py`

### 5.1 继承关系

`PXDPOModel(ProtenixDesign)`，直接继承 PXDesign 模型类。

好处：

1. 条件嵌入 (`get_condition_embedding`) 与原模型一致
2. 扩散去噪器 (`diffusion_module`) 直接复用
3. 减少“自定义网络结构导致输入不兼容”的风险

### 5.2 训练用 diffusion surrogate loss

定义了 `diffusion_recon_loss(...)`：

1. 抽样噪声尺度 `sigma`（来自 PXDesign scheduler）
2. 构造 `x_t = x0 + sigma * noise`
3. 调用 `self.diffusion_module(...)` 得到 `x_denoised`
4. 计算原子 MSE，并按 mask/权重聚合

支持：

- `atom_mask`（无效原子不计入）
- `binder_mask` 加权（binder/non-binder 不同权重）
- `fixed_noise/fixed_sigma`（让 chosen/rejected 在同噪声条件下比较）

---

## 6. DPO 损失定义

实现文件：`losses.py`

设扩散重建损失为 `L`（越小越好），分数定义为 `score = -L`。

对每个样本：

- `policy_delta = L_bad - L_good`
- 若有 reference：`logits = beta * (policy_delta - ref_delta)`
- 若 reference_free：`logits = beta * policy_delta`

损失：

- `loss = -log(sigmoid(logits))`

这与你在 `test.py` 的思路保持一致，只是扩展了 reference 分支与 metrics 输出。

---

## 7. 训练管道（train.py）

实现文件：`train.py`

### 7.1 训练步骤

1. 读取 Hydra 配置
2. 构建 PXDesign 配置（通过 `pxdesign.utils.infer.get_configs`）
3. 初始化 policy 模型并加载 checkpoint
4. 复制一份 frozen reference 模型（默认开启）
5. 根据 `model.trainable_modules` 冻结/解冻参数
6. 构建 dataloader（可按 target 分组）
7. 训练循环中：
   - 前向 policy good/bad
   - 前向 reference good/bad（`no_grad`）
   - 计算 DPO loss + 可选 SFT 项
   - 反向、梯度裁剪、优化器 step
8. 定期保存 checkpoint，结束时保存 `final.pt`

### 7.2 当前默认训练策略

- 参考模型：固定（`reference_free: false`）
- 损失区域：binder 加权（`use_binder_weight: true`）
- 每 step 单一 target：`group_by_target: true`
- AMP：按 `training.amp + training.precision` 控制

---

## 8. 关键配置项说明

### 8.1 `configs/config.yaml`（训练）

- `data.pairs_jsonl`: pair 索引文件
- `data.group_by_target`: 是否单 target step
- `training.batch_size`: 每 step 的 pair 数（同一 target 内）
- `training.grad_accum_steps`: 梯度累积
- `training.num_noise_levels`: sigma 抽样步数
- `dpo.beta`: DPO 温度系数
- `dpo.reference_free`: 是否不用 reference
- `dpo.use_binder_weight`: 是否使用 binder 掩码加权
- `dpo.binder_weight/non_binder_weight`: 区域权重
- `model.trainable_modules`: 可训练模块前缀

### 8.2 `configs/make_pairs.yaml`（pair 生成）

- `input.complex_dir`: CIF 文件目录
- `input.score_json`: 分数字典
- `pairing.w_ptm/w_iptm`: 排序权重
- `pairing.pairs_per_target`: 每 target 生成 pair 数
- `make_pairs.binder_chain`: 指定 binder 链（可空）
- `make_pairs.use_msa`: 是否在特征化时启用 MSA

---

## 9. 与 PXDesign 对齐与不报错保证点

当前实现做了以下对齐保障：

1. **输入特征来源对齐**  
   使用 `InferenceDataset.process_sample_dict/process_one` 构造 `input_feature_dict`
2. **模型调用对齐**  
   训练模型继承 `ProtenixDesign`，复用其条件嵌入与扩散模块
3. **张量类型对齐**  
   统一 `coord/mask` 为 `float32`，模型端按 device + AMP 自动处理
4. **batch 结构可控**  
   当 binder 长度一致时支持 stack；若不一致会在 stack 阶段暴露问题，避免 silent bug
5. **target 级采样**  
   通过 `TargetBatchSampler` 强约束一个 batch 单 target

---

## 10. 运行示例

### 10.1 先生成 pairs

```bash
python -m PBD-RL.make_prefer_pair.build_pairs \
  input.complex_dir=/path/to/complex \
  input.score_json=/path/to/scores.json \
  output.out_dir=/path/to/pairs \
  pairing.w_ptm=0.2 pairing.w_iptm=0.8 \
  pairing.pairs_per_target=64
```

### 10.2 再启动训练

```bash
python -m PBD-RL.train \
  data.pairs_jsonl=/path/to/pairs/train_pairs.jsonl \
  data.group_by_target=true \
  training.batch_size=4 \
  training.epochs=20 \
  training.lr=5e-6 \
  dpo.beta=0.3
```

---

## 11. 来自 boltzgen 可借鉴的增强策略（建议）

结合 `boltzgen` 代码，可优先借鉴这些策略（与当前实现兼容）：

1. **共享噪声/时间步对比（已做）**  
   chosen/rejected 用同一个噪声条件，降低 DPO 方差
2. **梯度裁剪（已做）**  
   抑制扩散训练中的梯度爆炸
3. **AMP + bf16/fp16（已做）**
4. **参数子模块微调（已做）**  
   先只训 `diffusion_module`/条件嵌入，减少不稳定
5. **随机旋转平移增强（建议下一步加入）**  
   `PXDesign` 采样中已有 `centre_random_augmentation` 思路，可在训练 `x0` 上增加等变增强
6. **EMA（建议下一步加入）**  
   boltzgen 使用 EMA callback，可提升推理稳定性
7. **学习率 warmup + 衰减（建议下一步加入）**  
   参考 boltzgen 的 AF3 风格 scheduler
8. **异常 batch 跳过与健壮日志（建议下一步加入）**  
   boltzgen 对坏样本容错较完善，可迁移

---

## 12. 当前假设与注意事项

1. chosen/rejected 坐标对应的**原子顺序一致**（这是 DPO 直接比较的基础）
2. 一个 pair 内 chosen/rejected 共享同一 `input_feature_dict`（同一 target 条件）
3. `binder_chain` 推断逻辑默认“非 target 的唯一链”；若复合物链多，请显式配置
4. 目前仍是 MVP 训练版，尚未引入 DDP、EMA、复杂 scheduler、在线验证采样

---

如果你愿意，我下一步可以继续补两件事：

1. 给 `make_pairs` 增加“hard negative”采样策略（按分数差控制 pair 难度）
2. 给 `train.py` 加入 EMA + warmup/decay scheduler（对齐 boltzgen 常见训练 trick）

