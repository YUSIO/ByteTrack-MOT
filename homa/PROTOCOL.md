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
