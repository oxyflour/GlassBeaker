- 先总结下 tasks/09-nijika-train.md 最新的数据集情况和误差
- brainstorm 下一步如何改进模型

## 数据集情况总结 (2026-04-16)

**当前数据集 (`tmp/antenna-dataset-2400-v2`)：**
- **总样本数**: 6600 (原有 2400 + 追加 4200)
- **生成状态**: 4200 个新增样本全部成功，0 失败
- **nib 边随机化**: 24 种唯一的 nib-side 组合模式

**当前最佳误差表现 (2400-sample v2)：**
- Validation RMSE: 0.13878
- Validation dB MAE: **7.09 dB**
- Validation dB RMSE: 11.06 dB
- Best Epoch: 223/300

**关键发现：**
1. **模型容量不是瓶颈** - 更深的 3-layer MLP 和 split decoder 实验均表现更差
2. **数据多样性是核心** - nib-side 随机化增加了任务难度，但提升了真实泛化能力
3. **损失函数重加权无效** - coupling-weight 和 notch-weight 实验均失败
4. **数据扩展收益递减** - 6600-sample 训练 epoch 1 指标与 2400-sample 相似

## 下一步计划：物理约束损失 (Physics-Informed Loss)

选择方案 C，在损失函数中加入物理约束来正则化模型：

1. **互易性约束 (Reciprocity)**: S_ij = S_ji
   - 当前模型输出已经是上三角镜像，但可以在 loss 中显式惩罚非对称性
   - 权重: `--reciprocity-weight`

2. **无源性约束 (Passivity)**: |S| <= 1
   - 惩罚预测幅度超过 1 (0 dB) 的区域
   - 权重: `--passivity-weight`

3. **因果性/平滑性约束 (Causality/Smoothness)**:
   - 频域导数连续性惩罚
   - 已在现有 loss 中有 `--smooth-weight`，保持不变

实施步骤：
1. 在 `training_utils.py` 中添加物理约束 loss 计算函数
2. 在 `train.py` 中添加 CLI 参数控制各约束权重
3. 从 6600-sample 数据集继续训练，对比 baseline
4. 验证指标: dB MAE, RMSE, 以及物理约束违反程度

## 进展更新 (2026-04-16)

**物理约束 loss 实现完成：**
- 添加了 `reciprocity_loss()`: 惩罚 S_ij ≠ S_ji 的非对称性
- 添加了 `passivity_loss()`: 惩罚 |S| > 1 的违反无源性区域
- 更新了 `composite_loss()` 整合物理约束
- 添加了 CLI 参数 `--reciprocity-weight` 和 `--passivity-weight`
- 添加了验证指标 `val_reciprocity_mse` 和 `val_passivity_mse`

**Smoke test 结果：**
- 代码运行正常
- `val_reciprocity_mse`: ~2.6e-18（几乎为 0，模型本身已满足互易性）
- `val_passivity_mse`: 0.0（没有违反无源性约束）

**正在进行的实验：**
- 命令: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-physics-6600 --model-kind structured_pair_spectral_head --epochs 300 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10 --reciprocity-weight 0.01 --passivity-weight 0.01`
- 目标: 对比 baseline (7.09 dB MAE) 与物理约束版本
- 权重选择: reciprocity=0.01, passivity=0.01（较轻的权重避免过度正则化）
- 状态: 训练中
