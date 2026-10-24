# Detector-feedback migrations (2026-10-24)

This branch implements the PPTracker mechanism migration, not an author-code or score reproduction. EEC is implemented on the separately recorded Exp161 branch. Official ByteTrack association is retained. See the research record in MOT/experiments/exp160_pp_feedback_yolox_x.

## PPTracker source and missing details

Source: CVPRW 2025, Qin et al., https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/papers/Qin_PPTracker_Tracking_UAV_Swarms_with_Prior_Prompt_CVPRW_2025_paper.pdf . Sections 3.1-3.3, 4.1-4.3 and Figure 2 were checked against PDF text and the rendered method page.

Eq.1 is `P = softmax(Q K^T / sqrt(h)) V`. With N track tokens and L image patches this produces N x D, not an image. Eq.2/Figure 2 adds it directly to the input image but does not define the shape conversion. The term "spatial probability matrix" also refers to the post-V result, whose rows are not probabilities. We implement the displayed algebra, not that descriptive label.

Explicit implementation choices: quantize normalized xywh coordinates into four independent 1024-entry word embeddings; add a persistent-ID embedding (4096 entries; overflow raises, no modulo collisions); dimension 64; nonoverlapping RGB patches of size 16; single scaled dot-product attention, no undocumented transformer blocks. To complete the missing N x D -> image mapping, scatter prompt tokens back by A^T P / N and linearly decode nonoverlapping RGB patches. This bridge is OUR choice, not an author formula. Empty prompts bypass exactly. Zero-initialize the final projection for an identity start. Prompt is added to normalized image input before the YOLOX backbone, as Figure 2/section 3.3 suggest; it is NOT a stride-8 neck injection.

Inference begins with detector-only frame 1, whereas the author competition provides first-frame GT. Subsequent prompts use previous confirmed ByteTrack outputs in detector-box geometry. ID vocabulary resets per sequence. Training uses previous-frame TRAIN GT boxes/IDs (teacher forcing, training prior provenance unspecified in paper). No test annotation is read by inference. Raw image patch embedding sees normalized RGB tensors, rather than unspecified author pixel normalization.

## Shared pipeline

YOLOX-X depth1.33/width1.25, one class, input800x1440, conf.001/NMS.7. ByteTrack parameters match Exp141 run004. Keep matched detector boxes for outputs (one decimal) and prior observations; Kalman state and matching stay unchanged. Round online det xywh to 4 decimals and scores to 6 before association to match Exp141 cache precision. `np.float` -> `float` is only NumPy compatibility. The observation hook does not modify track decisions. Only one sequence runner may be active per process because upstream IDs are global.

Use `python -m feedback.infer --help` and `python -m feedback.train --help`. Both require explicit inputs/output directories. Inference refuses random feedback unless explicitly labeled preflight. Training uses only root/train, explicit sequence lists, standard YOLOX losses, SGD momentum.9/weight_decay1e-4, constant user-specified LR, fixed BN running statistics, no random augmentation and teacher-forced train history. These are shared migration choices, not either author's full recipe. All detector parameters including BN affine remain trainable. Epoch/last weights are saved; no validation-selected best or final score is claimed. PSMP-specific training is on Exp161. Formal fit/dev split, schedule, checkpoint selection and remote runs require a separately frozen experiment configuration.

The none arm controls extra training, not just an inference-time ablation of trained PP weights. Tests use reduced YOLOX only for structural speed and never label its results YOLOX-X accuracy.

A single-anchor postprocess boundary fix restricts squeeze to its class dimension; normal multi-anchor threshold/NMS behavior is unchanged.
