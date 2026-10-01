# sonar-representation-lab — initial protocol

Author-evaluated exploratory study. Created 2026-10-01 before detector or SSL training. No independent review has occurred.

## Question and scope

Does self-supervised adaptation to unlabelled sonar imagery produce reusable features that improve label-efficient fish detection and cross-location generalization? A negative measured contrast is valid; absent training is not a result. No species, biomass, tracking, counting, deployment or SOTA claim is planned.

This project is separate from AEON. No stopped reviewer, other project directories, historical jobs, reserved data or denied operations may be accessed or rerouted. Seeds 13 and 23 here refer exclusively to newly defined CFC experiments, never the blocked historical jobs.

## Data policy

Use CFC v1.1 grayscale `kenai.tar`, `coco_annotations_v1.1.zip`, `file_lists_v1.1.zip`, and the official shared clip metadata. Resolve downloads from the publisher each time, verify published MD5 and record local SHA-256. Keep full official train and validation clip boundaries and all negative frames. Use full `kenai-train.json`, not the publisher's frame-subsampled training annotations. Rank train clips by SHA-256 of `sonar-lab-subsets-v1:<clip_name>`; nested fractions 1%, 10%, 100% contain ceil(fraction * number of train clips). Subset identity is fixed across encoder types and seeds. Retain every annotation and frame in each selected clip.

SSL reads a separate image-only Kenai TRAIN manifest; it cannot access annotations, track IDs, category labels, Kenai validation or Channel imagery. Channel remains a held-out location, with no target adaptation. Initial acquisition inspection accidentally displayed a few Channel annotation/metadata headers together with the other metadata files; no Channel imagery, model output or scores were viewed. Preserve this exposure record. Final Channel evaluation follows a written freeze of all models, thresholds and preprocessing. No model repair after that evaluation may be called untouched.

Data storage: D:/sonar-representation-lab-data. E: had 65.6 GiB free, insufficient for the 41.05 GiB archive plus extraction and cache. D: had 297 GiB free. No other data is deleted. The publisher names the image archives `.tar`, but Kenai's observed magic bytes identify gzip compression; extraction must detect compression automatically. Exact expanded size is checked before extraction. Dataset record rights and publisher usage terms are recorded separately from the software's MIT license; a software license must not be represented as the dataset license.

## Models

Official DINOv2 ViT-S/14 with four registers, revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, official `dinov2_vits14_reg4_pretrain.pth` downloaded from Meta's public host. Inspect source before import; no remote installer scripts. Use native PyTorch attention without xFormers or custom CUDA extensions. Initial baseline and SSL start from the identical checkpoint; record its SHA-256.

Common detector: spatial patch grid from the final normalized transformer layer, resampled to feature levels at strides 8/16/32/64, learned common pyramid projections, Faster R-CNN RPN/ROI heads. The backbone keeps spatial features. All detectors use the same image size, feature interface, head parameter initialization for each seed, sampler, labelled clips, optimizer steps and inference rules. One fish category plus background. Random frozen same-architecture encoder is a diagnostic. Supervised fine-tuning of the published encoder is a separately named comparison, with the same head and detector budget.

Detector preprocessing: aspect-preserving resize to fit 448 x 448, top-left padding with black pixels, replicate the grayscale plane three times, ImageNet normalization. Repeated grayscale is an adapter, not three acoustic frequencies. Map detections back through the actual rounded x/y resize scales, clamp to the original image, remove degenerate boxes. No annotation is removed because it is small. Training has no geometric augmentation in this first block.

## Declared adaptation setting A (one configuration)

DINO-style cross-view teacher/student distillation, initialized from the published encoder. Two independently sampled global crops 224 x 224 (area scale 0.4–1.0) plus two local crops 112 x 112 (area scale 0.1–0.4), aspect ratio 0.75–1.333, bilinear resize; grayscale brightness/contrast jitter and Gaussian blur. No color hue/saturation, no solarization, no temporal objective, no assertion that adjacent frames depict the same fish. These are grayscale-compatible modifications of the official DINO crop/blur policy; their biological invariance is a hypothesis.

Official DINO projection head with 4096 output prototypes, 1024 hidden units and 256 bottleneck (resource-limited dimension changes declared). Student temperature 0.1; teacher temperature linearly warms from 0.04 to 0.07 over the first 200 optimizer steps. Cross-entropy excludes identical global views. Teacher requires no gradient and runs under no_grad. EMA momentum increases by cosine from 0.996 to 1.0; update only after a successful optimizer step. Center momentum 0.9, update on raw teacher logits aggregated across accumulated microbatches after computing targets with the previous center. Freeze projection output layer for first 100 steps. AdamW peak backbone/head LR 1e-5 with 100-step warmup and cosine decay to 1e-6, weight decay 0.04, gradient norm cap 3. BF16 autocast, batch 16, accumulation 1 (effective 16), 2000 successful updates. This replaces the initial batch 4 / accumulation 4 plan after pre-run profiling demonstrated lower runtime and acceptable memory; no BatchNorm is used, so the effective objective/budget is preserved. Export EMA teacher encoder. Record collapse indicators and weight changes, not only loss.

At most two adaptation configurations are permitted; only A is declared now. Implementation defects and interrupted/failed runs are logged, never erased. Do not search continually until SSL wins. Final-step encoder and final-step detector are used, so no best-seed or checkpoint cherry-picking. Any second adaptation configuration must be declared before execution and selected using Kenai validation only.

## Detector budget and sequence

Profile real data before a long run. Start seed 7, 10% clips, published frozen backbone; train then evaluate on the complete Kenai validation set. Complete genuine SSL A, then train a fresh matched head on exactly the same 10% clips. Fixed detector budget after real TRAIN profiling: 2000 AdamW updates, batch 8, accumulation 1 (effective 8), head LR 3e-4 with 100-step warmup and cosine decay, weight decay 0.01, fine-tuned backbone LR 1e-5. Batch 2 / accumulation 4 was changed before any research training: batch 8 was faster and both frozen and fine-tuned real profiles stayed within memory ceilings. Fixed-step sampling may not visit every available frame; save image exposure counts and preserve all available labels.

Before research training, priority is fixed as follows: paired frozen contrast at 10% for seeds 7, 13, 23; fine-tuned and random diagnostics with the predeclared seed 7; paired 1% and 100% extensions first for seed 7, then seeds 13 and 23 when viable. Diagnostic replication at seeds 13 and 23 is lower priority than label-fraction extensions and may be omitted within the initial budget. This follows real profiling and reserves compute for Channel; no metric-dependent choice is made. Record each attempted seed, missing cells, clip/frame/annotation/negative counts, actual frame exposures and mean/sample standard deviation across seeds. Seed 7 is the predeclared diagnostic seed, not a selected best seed.

## Evaluation

Use official COCO annotations and pycocotools bounding-box evaluator. AP50 primary; also AP50:95, complete COCO stats/output, precision/recall at IoU 0.5 and score 0.5, false positives per negative frame and fraction of negative frames with any FP. Keep raw predictions with original image/category IDs and xywh coordinates. Score threshold for saved detections 0.001, NMS 0.5, at most 100 detections/frame. Include zero-detection images in the full evaluator image set. Known-answer tests: perfect/missed/false-positive boxes, empty negatives, category and image identities, xywh/xyxy and rounded resize round trips. Verify real TRAIN geometry and checkpoint reload/resume behavior. Author execution is distinct from independently reviewed findings.

After freezing, download only grayscale `channel.tar`, verify it, evaluate all fixed candidate models in one held-out evaluation block and save predictions and full evaluator state. Do not select or repair models based on Channel. This is one held-out location, not the full CFC benchmark.

Channel structural metadata was read before training solely to reserve compute: 69 clips, 13,090 frames in its official v1.1 file list (the older clip metadata totals 13,159). No Channel pixel or model score was viewed. Optional fraction pairs require measured train/validation runtime plus a conservative reservation for all fixed Channel models to fit inside 23.8 hours, with admission also stopping at 20 hours already charged. This is runtime-based admission, never AP-based selection.

## Resources and reproducibility

Windows private `.venv`, pinned dependencies and official source revision; one training process. Keep owned RAM below 22 GiB and GPU usage below 10 GiB, account for existing desktop GPU usage. Record per-run elapsed GPU-active process hours (including profiling, inference and failures), GPU allocated/reserved peaks, process RSS and child RSS, step losses/LR, checkpoints, raw detections and metrics. User reported 100 GPU-hours remain overall; initial block cap stays 24 local GPU-hours and is deducted from that balance. No cloud purchase or public deployment. Abort before exhausting limits and preserve resume state. Ordinary training has no LLM-signature dependency.

## Source facts and comparator context

- [CFC v1.1 record](https://data.caltech.edu/records/g945x-41103) and [official guide](https://github.com/visipedia/caltech-fish-counting/tree/main/CFC): grayscale and alternative three-channel releases are distinct; official Baseline/++ AP50 values are context, not matched experimental baselines.
- [DINOv2 source](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) and [DINO training reference](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py): objective/EMA/centering sources. This project is domain adaptation of an existing representation, not training a foundation model from scratch or reproducing full DINOv2 pretraining.
- [ALDI (TMLR 2025)](https://aldi-daod.github.io/), [paper](https://arxiv.org/abs/2403.12029): CFC-DAOD adds target-domain adaptation data and uses a different protocol; target-adapted numbers are not source-only generalization results for this project.
