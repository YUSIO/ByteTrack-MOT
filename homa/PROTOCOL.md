# Exp042：YOLO11s/V2 上的 paper-guided HOMATracker

来源：Chu et al., Vision-based swarm tracking of multiple UAVs in air-to-air scenarios，DOI 10.1016/j.cja.2025.103558。2026-10-03 通过 Zotero B8VI5XHI / 6Y8UGKML 取得 15 页索引全文。无已确认的作者实现或权重。本实现不得称为官方实现或原论文数字复现。

## 论文依据与独立选择

- 论文：ResNet、128×128 crop、256×8×8 feature、宽度 1:2:1、2×2 patch / stride 2、embed 512，历史 self-attention，当前 cross-attention，ReLU MLP，分块点积，1×1 conv + ReLU；按历史帧 softmax 后按 tracklet 累积。
- 选择：ImageNet V1 ResNet50 至 layer3，再用 1024→256 的 1×1 conv。flattened part 维度为 2048/4096/2048；每块先投影到 256 维、8 个 heads、MLP hidden 512，再输出原 part 维度。采用 residual + LayerNorm，点积除以 sqrt(part_dim)。这些内部细节原文未明确，属于自选实现，不应推定为作者配置。
- 论文训练：30 epoch、SGD lr=.002 / momentum=.9 / decay=.0001、batch 8。实现用 8 个窗口梯度累积达成有效 batch 8，AMP，固定 backbone BN running statistics；无数据增强，gradient norm clip 10。seed=42。
- 监督：式 (22)(23) 的三个 part 对每个历史帧做同 ID 分类，缺失 ID 不计算 loss；另加一次 fused logits CE 以训练 conv fusion，原文未说明 conv 的训练路径。仅用 MOT-train GT crop 和轨迹 ID。loss 为 3×part mean + fused mean；这不是 ID classifier。
- 训练与验证：既有 Exp026 36/12 序列划分。T=8，训练采样 stride 8，验证 stride 16；仅当前帧向过去帧的因果关联。checkpoint 只按 validation loss 选择。
- M2DA：式 (9) 卡尔曼位置均值/协方差；式 (10) 检测 std=w/h 按印刷式实现，可能存在原文笔误。保留单步预测，按帧淘汰，Wasserstein 平方距离 **sum**；不是 Exp037 的 mean。kappa=.1 或 200、T=2 或 8 是已列明的 validation 网格。坐标为像素，kappa 单位为平方像素；论文未说明坐标尺度。
- MHA：第一阶段 M×S，cost=1-similarity，不再叠加检测分数。MPA+IoU 用 M×IoU。M 是按历史帧累积的 sum，可能大于 1；不擅自 clamp 或求平均。
- 生命周期：沿用固定 ByteTrack 的新轨阈值 .7、high>.6、low>.1 且 <.6、second-stage IoU、unconfirmed confirmation、lost buffer=30 和后验框输出；不声称还原论文未公开的生命周期。
- 缓存：Exp011 修正版 48 个 test cache 原样上传；validation 使用 Exp028 的 12 序列 cache。图像仅为 crop 输入，tracker 从不读取 GT。detector 不重训、不重导出。
- 四臂均保持 min_box_area=100、aspect_ratio=3.0；此历史 ratio 有 test 暴露，结论仅为工程对照。验证集前半段也被 detector 见过，关联器训练仍保持 36/12 序列隔离。
- 每个训练、网格配置与 test 执行是独立 run；失败保留。验证 20 runs，各臂仅选一次最优 HOTA→IDF1→较少 IDSW；test 固定执行四臂，无 test 调参。
- evaluator 复用原文件 SHA-256 b887a8f4a2bc70519a3d5ca9ca9ab088c0965b0e5b7979f0adb1f79c1f5456ff；motmetrics GT/result frame union，TrackEval HOTA。依赖版本另存。

## 执行

`python -m homa.train` 是训练入口，`python -m homa.track` 只读检测/图像，`python -m homa.evaluate` 独立读 GT。`homa.run` 拒绝已有 run 目录、记录准确命令/commit/退出码/日志/哈希。`homa.pipeline.sh` 顺序执行训练、验证、冻结 test；任意失败即停止，不自动覆盖或重试。

## 2026-10-03 数值处理修复

run_001 在 epoch 2 遇到非有限梯度并退出。修正 AMP 溢出处理：整次累积跳过 optimizer step，loss scale 减半，参数与 momentum 保持不变，记录跳步；连续 16 次溢出才终止。非有限 forward loss 仍立即失败。新 run 从相同 seed 的 ImageNet 初始化完整重训 30 epochs，不恢复失败 checkpoint；配置、数据划分、loss 和模型结构不变。pipeline 支持显式新起始 run 编号，拒绝覆盖已有目录。

## v2_repair：2026-10-03 实现审计后的修订

本节仅用于 `architecture=paper_v2` / `lifecycle=mha_v2`，旧 checkpoint 使用 `legacy_v1` 路径。v1 运行不可在 v2 代码上无条件重演，应 checkout 各 run 原 SHA。

### v3 可学习性修订

`architecture=residual_norm_v3` 在 v2 full-D head 上对 history/current 向量先执行无 affine 参数的 functional LayerNorm，再保留 `h + SA(h)` 和 `q + CA(q, enhanced_history)` 两条残差。head/fusion 保持 FP32、backbone AMP，all-frame/pair-weighted loss 与 MHA 生命周期不变。这是失败诊断支持的工程实现选择，不宣称已确认作者结构。

正式训练前 `homa.preflight_v3` 在三个训练序列窗口、seed 42/43、真实 AMP/8-window 累积/optimizer 路径上验证 32 次更新可降低 loss；不保存或继承预检权重。正式运行重新初始化。训练记录解析均匀 loss，连续三 epoch 相对均匀基线收益低于 .001 时主动失败，保留当前 checkpoint/logs；新尝试使用新 run。

- 删除外部 256-D bottleneck、residual/LayerNorm 和式 (4) 的额外 sqrt(D) 缩放。SA/CA 直接在各 part 的 2048/4096/2048 维进行，8 heads；ReLU MLP hidden=512。标准 attention 内部 QK scaling 保留。head/fusion 显式 float32，backbone 使用 AMP，避免未缩放点积的 fp16 溢出。
- 窗口内每帧轮流作为 query，其他帧为候选，排除同帧。按有效同 ID 跨帧 pair 累加三块 CE，再加 fused CE（权重 1），以 pair 数归一化。fused 监督仍为原文未明确的自选项；不是声称修复成了作者原 loss。训练可见窗口内双向 pairs，在线推理仅过去帧；train/val 的 36/12 序列边界不变。
- MHA 独立实现，原 ByteTrack 控制不改。轨迹置信度取最近匹配 detection 的 score（原文未定义，显式选择）；high 轨迹参与第一阶段，low 轨迹与未匹配 high 中上帧出现者参与 low-IoU 第二阶段。所有未匹配 high 立即出生，无额外 .7 阈值或 unconfirmed 阶段。轨迹观察窗口为 T=8，不再沿用 ByteTrack 的 30 帧 retained-lost 生命周期。第二阶段阈值 .5 保留为固定选择。
- 增加 `mha_iou` 控制分离生命周期效应。验证共 22 configs：byte/mha_iou/mpa_iou 各 2，m2da/homa 各 8。motion T=8、κ=.1 是原文形式参考；κ=200 属迁移变体，T=2 是单次预测控制。两者不得冒充原文参数或多帧收益。MPA T=8 有原文 UAVSwarm 消融依据。
- 从 ImageNet 初始化重训 30 epoch，不恢复 v1 权重；seed、SGD 与数据不变。只运行训练及 validation，`HOMA_RUN_TEST=0`。先核验 head 与 lifecycle，再决定新的冻结 test；不按已有 test 结果调参。
- 尚未确认作者代码；ResNet50 stage/projection、head 数、hidden size、融合监督、confidence 语义仍为明确的独立实现选择。本修订为验证修复假设，不能预先称为忠实复现成功。
