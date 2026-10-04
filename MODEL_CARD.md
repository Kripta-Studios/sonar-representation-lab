# Model card: sonar-representation-lab

Latest completed follow-up: The seed-7 image branch passes the practical screen. B initialization provides no label-efficiency improvement in the completed matched pilot. Primary AP50 change +1.7295 percentage points. See the [localization follow-up](#localization-follow-up--completed-author-execution) below for new checkpoints, evaluator v2, resource charges and omissions. Earlier dated sections retain their historical results.

Author-evaluated exploratory research. Updated through completion of the fixed Channel block at **2026-10-03 17:36 UTC**. No independent review or operational deployment validation has occurred. This card covers actual locally trained artifacts, with hashes verified against the frozen roster. [README.md](README.md) describes the architecture and commands; [RESULTS.md](RESULTS.md) provides the full training/results report; [PROTOCOL.md](PROTOCOL.md) preserves the experimental policy.

## Artifact scope and completion

There are **six completed adapted encoders** (A seeds 7/13/23, B, B-CONTROL and C-MOTION) and **fifteen completed detector fits**, all with full Kenai validation. The existing random checkpoint scored 9.5298% AP50 without retraining. At the selected 672 resolution, published/A/B/control/motion frozen detectors score 26.9978%/20.7433%/20.5470%/20.5152%/20.7500% AP50. Matched full fine-tuning scores 49.7605% from published initialization versus 48.5271% from B. Neither B nor C passes the predeclared replication rule. All seven fixed seed-7 checkpoints completed the source-only Channel block once; results follow below. No 1%/100% detector comparison has been completed.

The primary published/adapted contrast is complete at 10% labelled clips for seeds 7, 13 and 23. Adaptation reduced mean AP50 from 15.6714% to 11.3495%, a paired change of -4.3219 percentage points. The separately labelled seed-7 supervised fine-tuning control reached 45.4791% AP50. It uses labels to update the encoder and is not an SSL result.

## Model architecture

The encoder is official DINOv2 ViT-S/14 with four registers, 12 transformer blocks, width 384 and six attention heads. Detection retains the final normalized patch grid: 32-by-32 at 448 input or 48-by-48 at the selected 672 input. Four bilinearly resampled levels feed learned 96-channel projections, a Faster R-CNN region-proposal network, multi-scale ROIAlign and a two-layer 1,024-unit ROI head. These levels resample one final grid; they are not four backbone stages. There is one foreground category, fish, plus background.

Published and adapted primary detectors freeze the encoder while training the pyramid/RPN/ROI components. The random diagnostic freezes the same architecture with random encoder weights. The fine-tuned control starts from published weights and trains both encoder and detector head. Seeds determine reproducible sampling and matched head initialization; they are not selected by validation performance.

Configuration A adapts the published encoder through a DINO-style global-token teacher/student loss, using two 224-pixel global and two 112-pixel local crops, six cross-view loss terms, detached teacher targets, EMA updates, centering and the recorded temperature schedule. The final EMA teacher supplies the exported encoder. A does not include direct patch-level distillation or temporal consistency. Its exported patch features remain available for downstream detection, but their usefulness must be assessed by detection results.

## Inputs and outputs

Input is an official CFC v1.1 grayscale frame. Preprocessing aspect-preserving resizes it onto the checkpoint's fixed 448-by-448 or 672-by-672 canvas with black padding to the right/bottom, repeats the grayscale plane three times, and applies ImageNet normalization. The three channels do not represent different acoustic frequencies. Actual rounded x/y scale factors govern box transformations. Resolution is part of checkpoint/cache identity; 672 jobs read original JPEGs and reject a 448 cache.

The detector produces fish boxes and confidence scores. Saved predictions retain official image IDs, category ID 1, original-image coordinates and COCO `xywh` boxes. Filtering retains scores at least 0.001, applies IoU-0.5 NMS and limits output to 100 boxes per image. Precision/recall reporting uses score 0.5 and IoU 0.5; AP evaluates the saved low-threshold detections. Neither output is a species label, track ID, count estimate or biomass estimate.

## Published initialization

- Source revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8` in the official DINOv2 repository.
- Published checkpoint: [dinov2_vits14_reg4_pretrain.pth](artifacts/pretrained/dinov2_vits14_reg4_pretrain.pth), 88,291,785 bytes.
- Published SHA-256: `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`.
- Original publisher: [Meta checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth).

The factory checks both source revision and checkpoint identity. Source was inspected before use. No remote setup script or third-party model installer was blindly executed.

## Adapted encoder inventory

The table below preserves configuration A's three completed encoders. Configuration B's separately trained seed-7 artifact is documented at the end of this card.

Each encoder completed 2,000 genuine SSL updates with effective batch 16, or 32,000 frame presentations. Unique TRAIN images were 29,017 / 29,073 / 28,977 for seeds 7 / 13 / 23 respectively. All three exports passed the saved author audit for finite tensors, exact equality to final EMA teacher weights and TRAIN-only image exposure.

| Seed | Encoder artifact | Bytes | SHA-256 |
|---:|---|---:|---|
| 7 | [Final EMA encoder](artifacts/ssl-A-s7/encoder.pt) | 88,297,055 | `15b3356ef61f163a955cefd240b04555635b104f7b222dcd408fdb783eaca210` |
| 13 | [Final EMA encoder](artifacts/ssl-A-s13/encoder.pt) | 88,297,055 | `243cef3824e30edefce9c7a507a6dc3af25e0b161d2f88e8dd2ade4dbd66fccb` |
| 23 | [Final EMA encoder](artifacts/ssl-A-s23/encoder.pt) | 88,297,055 | `41345e0e978c880c3d8e0371379b35cee3f85df5080dc52f29bebbf5a1ad122d` |

An `encoder.pt` file contains the encoder state, experiment configuration, successful-update count and export identity. The adjacent `checkpoint.pt` contains the complete resume state: teacher/student encoder and head weights, center, optimizer, RNG and exposure counts. The distillation head is not part of the downstream fish detector.

## Detector inventory

All eight detectors below trained for 2,000 updates with effective batch 8 on the same fixed 10% clip subset. Checkpoints contain model and optimizer state, configuration, update count, RNG state and per-image exposure counts. Their sizes include resume state and should not be interpreted as encoder-only model sizes.

| Kind | Seed | Checkpoint | Bytes | SHA-256 |
|---|---:|---|---:|---|
| published | 7 | [Weights](artifacts/det-published-f010-s7/checkpoint.pt) | 165,744,681 | `ae87feaa719f90c59c38cb7221937a8512d058ce30f7670e9d48a4c5c24bcf3c` |
| adapted | 7 | [Weights](artifacts/det-adapted-f010-s7/checkpoint.pt) | 165,744,873 | `346ca4ca64a64bf05058145b0458cc4156aa476139ec3995c7a2b3e938fb46aa` |
| published | 13 | [Weights](artifacts/det-published-f010-s13/checkpoint.pt) | 165,744,553 | `f29574bc80fd5dc4f6a7134d4e10558adb91a9e2d38926991a908e3cf0d28cfb` |
| adapted | 13 | [Weights](artifacts/det-adapted-f010-s13/checkpoint.pt) | 165,744,745 | `81e6729c1e5c1578dc1787d6124cbd75ac8fa4c9249c7f8b48114e492441418b` |
| published | 23 | [Weights](artifacts/det-published-f010-s23/checkpoint.pt) | 165,744,809 | `8a53010e9f114bb202b841f48ca01d893aaf8c37aadd3c750aa67b83f33adb06` |
| adapted | 23 | [Weights](artifacts/det-adapted-f010-s23/checkpoint.pt) | 165,745,001 | `9dc3d66ec8fd9484edd5496530235fd1f99f847a9a3b8bb78f36536a50329566` |
| finetune | 7 | [Weights](artifacts/det-finetune-f010-s7/checkpoint.pt) | 342,353,743 | `89971bfb5cb9d069e7f4b822cdbe8e9c10a36161f7ef0f50204f8a2b497896dd` |
| random | 7 | [Weights](artifacts/det-random-f010-s7/checkpoint.pt) | 165,745,001 | `3b01ee3a30427d9fd0b69ea27095b7146fab2041a9433eadf040426154c9f751` |

These SHA-256 values identify the files present at this report snapshot. Recomputing a hash does not replace an optimizer-resume test or finite-tensor audit. The historical consolidated audit covers three SSL exports and five then-completed detectors; later artifacts have their own training, exposure, resource and evaluation records. The exact audit scope is documented in RESULTS.md.

## Evaluation results

| Kind | Seed | AP50 (%) | AP50:95 (%) | Evaluation state |
|---|---:|---:|---:|---|
| published | 7 | 15.2853 | 4.0586 | Complete, 30,454 Kenai validation frames |
| adapted | 7 | 10.6641 | 2.4921 | Complete, 30,454 Kenai validation frames |
| published | 13 | 15.6095 | 3.9548 | Complete, 30,454 Kenai validation frames |
| adapted | 13 | 11.3871 | 2.8070 | Complete, 30,454 Kenai validation frames |
| published | 23 | 16.1196 | 4.3321 | Complete, 30,454 Kenai validation frames |
| adapted | 23 | 11.9974 | 2.8590 | Complete, 30,454 Kenai validation frames |
| finetune | 7 | 45.4791 | 14.8833 | Complete, 30,454 Kenai validation frames |
| random | 7 | 9.5298 | 2.5823 | Frozen random encoder; trained head |

All listed metrics are author-computed exploratory measurements from saved detections and pycocotools. The primary paired AP50 changes for seeds 7/13/23 are -4.6212 / -4.2223 / -4.1222 points. Sample standard deviations of AP50 are 0.4206 points for published and 0.6674 for adapted. These seeds share the same dataset and labelled clip subset, so their variation does not quantify location uncertainty.

At the fixed score-0.5 operating point, fine-tuning obtains 67.6499% precision and 44.6283% recall, with 538 false positives over 20,529 negative frames. The frozen models have much lower recall. AP50 improvements and the operating-point false-positive tradeoff must be considered together. Complete per-seed confusion counts and negative-frame metrics are in RESULTS.md.

The fixed Channel evaluation completed after the seven-model freeze. The initial metadata inspection exposed a few annotation/clip headers; the journal preserves that exposure. The official grayscale archive was acquired after freezing, verified and extracted. No checkpoint, threshold or preprocessing setting was selected or repaired using Channel scores. Future Channel use must disclose this completed exposure.

## Training data and label policy

Official Kenai TRAIN contains 482 clips, 162,198 frames and 132,010 boxes, including 92,544 negative frames. The fixed 10% subset contains 49 clips, 16,859 frames, 11,728 boxes and 10,705 negative frames. Every annotation in each selected clip remains available. Nested clip-level manifests preserve official TRAIN/validation boundaries; adjacent frames are never randomly divided between those splits.

The 2,000-step detector budget samples 16,000 presentations with replacement. It does not traverse every available image. The unique image counts are 10,356 for seed 7, 10,264 for seed 13 and 10,313 for seed 23 in the primary pairs. The random and fine-tuned seed-7 controls also report 10,356. Fine-tuning does not receive a larger labelled subset than the frozen comparison.

SSL reads a separate image-only TRAIN manifest. It has no access to fish boxes, classes, track IDs, validation images or Channel images. The training data are public CFC grayscale data acquired for this project; historical AEON models, environments, guards, reviewer records, reserved data and blocked jobs are outside this study.

## Reuse and reproduction

Use the project-private Python environment and pinned dependencies. The common CLI reads the full checkpoint, reconstructs the detector and applies the recorded preprocessing. For example, this is the command to evaluate the completed fine-tuned detector on Kenai validation into a separate output directory:

```powershell
.venv/Scripts/python.exe src/train_detector.py predict --checkpoint artifacts/det-finetune-f010-s7/checkpoint.pt --out artifacts/rescored-inference-finetune-s7
```

The example is a reproduction command, not a run performed during this documentation update. To recompute metrics without inference or GPU use, use `src/evaluate.py` with the saved prediction JSON and official validation manifest, as shown in README.md. Preserve the original prediction and evaluator files.

The training CLI accepts `--resume` only with matching configuration/source/protocol identities. A checkpoint that has not completed its declared budget cannot be final-evaluated by the standard predictor. Profile checkpoints and staged implementation drafts are not research model artifacts.

Retain the published/adapted initialization identities, complete label manifests, input resolution, head definition, confidence/NMS rules and evaluator configuration when making a comparison. Changing these settings creates a new experiment; the scores above do not transfer to that configuration.

## Intended use and limitations

The artifacts support local research on sonar representation transfer and fish bounding-box detection. They have not been validated for field deployment, safety-critical decisions, species identification, tracking, automated fish counting or biomass estimation. No Marine Instruments validation or SOTA claim is made.

The main scientific result is negative for configuration A under one matched 10%-label setting. Cross-location generalization and the full label-efficiency curve remain unanswered. The high fraction of small fish, very low AP75, limited sampled exposure, one fixed labelled subset and single-seed fine-tuning control constrain the interpretation. A good-looking attention map or nonzero weight change does not establish detection quality.

The observed source-only protocol differs from domain adaptation methods that consume unlabelled target-site images. Future changes after a Channel evaluation must disclose prior target exposure. Missing or unscored models receive no inferred performance value.

## Terms and provenance

The downloaded CFC record declares MIT rights; publisher terms and archive provenance are recorded locally. DINOv2 uses its official Apache-2.0 source/checkpoint terms. These data/model terms are distinct from the project code's license. Follow the respective upstream terms when redistributing weights or data.

Source and dataset links, exact acquisition commands, the official grayscale-versus-alternative representation distinction, and the dated comparator check are preserved in README.md, PROTOCOL.md and RESULTS.md. This card describes author-evaluated local artifacts and does not certify independent review.


## Continuation artifact: published frozen, 672, seed 7

[Detector weights](artifacts/det-published-r672-f010-s7/checkpoint.pt), SHA-256 `caeb86511dadc2aaf75d83f29e59a0df5e0fccb42a288ee9c8279afeb5d0544d`, completed 2,000 updates and 16,000 presentations across the same 10,356 unique frames as the original seed-7 detector. [Artifact checks](artifacts/published_resolution_training_identity.json) confirm identical full image-exposure counters, matched training settings and an unchanged published encoder in both 448 and 672 checkpoints. The exact declared source is archived beside the new checkpoint. Training charged 0.5961025 GPU-process hours. Full Kenai validation subsequently scored 26.9978% AP50 and 7.6219% AP50:95, charging 1.08175333 hours. Saved detections and complete COCO output are in its `val` directory.

## Continuation artifact: A frozen, 672, seed 7

[Detector weights](artifacts/det-adapted-r672-f010-s7/checkpoint.pt), SHA-256 `38e8aa935e5007da956434fb037fcffa8fb0b0d81a85c8321980e683edf92beb`, completed 2,000 updates and 16,000 presentations. [Four-arm checks](artifacts/resolution_training_identity.json) confirm identical exposure counters for published/A at 448/672 and unchanged frozen encoders. This run reuses the existing A encoder; no new A SSL was performed. Training charged 0.52202694 GPU-process hours; full validation charged 0.85802972 hours. AP50 is 20.7433%, AP50:95 5.2761%, and AP75 0.8053%. Raw detections and complete evaluator output are in its `val` directory. The matched published result is stronger.

## Continuation artifact: spatial B encoder, seed 7

[Final EMA encoder](artifacts/ssl-B-s7/encoder.pt), 88,298,399 bytes, SHA-256 `972ff06e32e7ce59b2b7625c6254f20c92505da76a692193630134d187ab3954`, completed 2,000 SSL updates at effective batch 16. Its 32,000 presentations cover 29,017 unique image-only Kenai TRAIN frames. [Artifact verification](artifacts/ssl-B-s7/artifact_verification.json) confirms exact EMA export, finite training modules, distinct global/patch centers, and identical full exposure counters, sampler state and final CPU augmentation RNG to A seed 7. Encoder relative-L2 change from published initialization is 0.0037022537; this is execution evidence, not a detection metric.

B adds same-crop masked-patch distillation with separate teacher/student patch heads and center to A's unchanged CLS path. Its 102-of-256 patch masks use an independent saved RNG; teacher targets stop gradients and follow the declared centering/temperature/EMA schedule. SSL crops remain 224/112 despite the downstream resolution choice. It is an iBOT-inspired sonar adaptation, with uniform masks and extra unmasked CLS passes, not an exact reproduction or an equal-compute ablation of A.

[Complete resume checkpoint](artifacts/ssl-B-s7/checkpoint.pt), 444,075,857 bytes, SHA-256 `4c1cadace072f13d96eefe89e9774e94a0e7f94152faa3b145ea050a0d40574b`, contains student/teacher encoders, both head pairs, both centers, optimizer, RNG/mask-generator states and exposures. Training charged **1.88150167 GPU-process hours**; physical GPU peak was 5,270,142,976 bytes and owned RAM peak 2,062,827,520 bytes. Exact loaded-state tests passed; bitwise training trajectories are not claimed. Source and protocol are archived beside the checkpoint. The fresh matched frozen detector has completed training and full validation, scoring 20.5470% AP50. This does not establish a frozen-feature benefit over the published encoder.

## Continuation artifact: B frozen detector, 672, seed 7

[Detector weights](artifacts/det-b-r672-f010-s7/checkpoint.pt), SHA-256 `a84d2616e454e71f377815d2d8c354874ac371adcbcb25fca4fc0a879a1a9840`, completed 2,000 updates and 16,000 presentations on the fixed 49 labelled TRAIN clips. Full exposure counters match published/A at 672; all encoder tensors remain identical to the final B initialization. The artifact check is `artifacts/spatial_B_frozen_training_identity.json`. Training charged 0.59666222 GPU-process hours. Full Kenai validation scored AP50 20.5470%, AP50:95 5.4390% and AP75 1.4190%, with 1,115,719 saved detections. Precision/recall at score 0.5 are 59.7426%/16.0153%; 79 false positives occur on 20,529 negative frames. Validation charged 1.04835917 GPU-process hours. B fails the frozen replication screen against published features. Its completed Channel result is 9.3323% AP50 and 2.5926% AP50:95.

## Continuation artifact: B full fine-tuning, 672, seed 7

[Detector weights](artifacts/det-b-finetune-r672-f010-s7/checkpoint.pt), SHA-256 `b9b2eea92e41a0901a6899a1f1236c38174530e07d07d4b3c866ec7e6b3c0d8f`, completed 2,000 updates at 2026-10-03 07:24:57 UTC. Full per-image counters exactly match B frozen: 16,000 presentations, 10,356 unique frames from the same 49 clips. Artifact checks confirm that 175 encoder tensors changed from B initialization, all model tensors are finite and all active head/backbone AdamW step counters equal 2,000. Encoder relative-L2 change is 0.0032787772; this verifies execution, not detection quality. Resume state and exact source/protocol snapshot are saved beside the checkpoint.

Training charged **0.82532972 GPU-process hours**, with physical GPU peak 7,138,705,408 bytes and owned RAM peak 2,541,084,672 bytes. Full Kenai evaluation completed with AP50 48.5271%, AP50:95 16.3736% and AP75 6.4977%, retaining 615,715 detections. At score 0.5, precision/recall are 69.2063%/46.9085%, with 527 false positives on 20,529 negative frames. Evaluation charged 0.95630639 GPU-process hours. The matched published reference scored 49.7605% AP50 and 16.6518% AP50:95. B loses 1.2334 AP50 points and 0.2781 AP50:95 points; the predeclared fine-only replication screen fails.

## Continuation artifact: published full fine-tuning, 672, seed 7

[Detector weights](artifacts/det-finetune-r672-f010-s7/checkpoint.pt), SHA-256 `06fa8ffb31baa3208a249d46e6c2e96e69c8d66dde598134d954f11e9dd93bb5`, completed 2,000 updates at 2026-10-03 09:19:29 UTC. The [paired check](artifacts/B_published_finetune_training_identity.json) verifies identical full per-image counters, label manifest, resolution and optimization budgets against B full fine-tuning. All active optimizer counters equal 2,000 and 175 encoder tensors changed from published initialization. The checkpoint contains complete resume state and has an adjacent source snapshot and learning curve.

Training charged 0.94048194 GPU-process hours. Physical GPU peak was 8,817,475,584 bytes and owned RAM peak 2,538,975,232 bytes. Full Kenai validation scored AP50 **49.7605%**, AP50:95 **16.6518%** and AP75 **6.3982%**, with 661,598 saved detections. Precision/recall at score 0.5 are 69.7005%/47.9273%; 596 false positives occur on 20,529 negative frames. Evaluation charged 1.02910167 GPU-process hours. This is the stronger initialization in the matched seed-7 fine-tuning contrast; no multi-seed or cross-location conclusion follows yet.

## Continuation artifact: paired B-CONTROL encoder, seed 7

[Final EMA encoder](artifacts/ssl-control-s7/encoder.pt), 88,299,295 bytes, SHA-256 `21b39048540f62de810453ea68233d7f80acdb4bcd6d539a98aa1353c573d7d4`. [Complete resume checkpoint](artifacts/ssl-control-s7/checkpoint.pt), 444,181,808 bytes, SHA-256 `7e05b28e1ee0b5943194c87d7a4dd6a5267bedb883f38919923526748f2a51f0`. This is 1,000 spatial continuation updates from B's complete seed-7 parent, using 8,000 verified adjacent TRAIN pairs/16,000 frame presentations and 15,218 unique frames. It adds no dynamic loss. Its final teacher export, finite states, B ancestry and inherited optimizer-counter advancement passed author verification. Training charged 0.58110694 GPU-process hours. Its matched frozen detector completed full Kenai validation at 20.5152% AP50; details follow.

## Continuation artifact: B-CONTROL frozen detector, 672, seed 7

[Detector weights](artifacts/det-control-r672-f010-s7/checkpoint.pt), SHA-256 `2c032b69d1457d87205ab1e09d9a2ec9604d7ad813ae09e3c945ba2b58881843`, completed 2,000 updates at **2026-10-03 11:36:45 UTC**. The [artifact check](artifacts/control_frozen_training_identity.json) verifies identical full image-exposure counters against published and B frozen: 16,000 presentations, 10,356 unique frames, the same complete labelled clips and optimization settings. All encoder tensors remain bitwise identical to the control initialization. Training charged **0.58559028 GPU-process hours**, with physical GPU peak 3,485,466,624 bytes and owned RAM peak 2,568,900,608 bytes. Resume state, source snapshot and learning curve are saved. Full Kenai validation completed at 2026-10-03 12:26:51 UTC: AP50 20.5152%, AP50:95 5.3663%, AP75 1.1251%; precision/recall at score 0.5 are 61.7856%/15.0342%. There are 66 false positives on 20,529 negative frames. All 1,270,528 detections and complete evaluator output are saved. Evaluation charged 0.82862833 GPU-process hours. Extra spatial continuation alone did not improve over B's 20.5470% AP50.

## Continuation artifact: C-MOTION encoder, seed 7

[Final EMA encoder](artifacts/ssl-motion-s7/encoder.pt), 88299295 bytes, SHA-256 `5ffa15b1b89454ffce98462e69d5089808df4499b1603d4f29591df18481d6f8`. [Complete resume checkpoint](artifacts/ssl-motion-s7/checkpoint.pt), 446841179 bytes, SHA-256 `dabf9240484cbc20e65f8c3458494c912aa802a2a99386843f90e074943b1146`. The fit completed 1,000 continuation updates at **2026-10-03 13:15:17 UTC**, starting from the same B parent as B-CONTROL. The paired artifact check verifies exact image-exposure, all RNG, mask-stream, parent and shared-setting identity: 16,000 frame presentations and 15,218 unique TRAIN frames. The final teacher export is exact; all training modules are finite and global/patch centers remain distinct.

This is a DISReg-inspired sonar auxiliary with explicitly declared variance/covariance regularization, not a full MotionJEPA reproduction. The dynamic coefficient was fixed at 0.02524451906202984 by TRAIN-only calibration. Training charged **0.80612389 GPU-process hours**; physical GPU peak was 5,624,561,664 bytes and owned RAM peak 2,868,408,320 bytes. The dense encoder requires one frame at downstream inference. Its matched detector has completed full validation; the motion screen is negative. No detection gain is claimed from SSL losses or feature diagnostics.

## Continuation artifact: C-MOTION frozen detector, 672, seed 7

[Detector weights](artifacts/det-motion-r672-f010-s7/checkpoint.pt), SHA-256 `acd910a2fe3b1363eaa5c76b8cff8ecde47207ee42365cd46dba83229dffb97a`, completed 2,000 updates at **2026-10-03 13:47:10 UTC**. The [matched artifact check](artifacts/motion_frozen_training_identity.json) verifies exactly equal full image-exposure counters against published, B and B-CONTROL at 672: 16,000 presentations and 10,356 unique frames from the same complete 49 clips. Encoder weights remain bitwise identical to the exported C-MOTION initialization. Training charged **0.51806861 GPU-process hours**; physical GPU peak was 3,484,418,048 bytes and owned RAM peak 2,569,003,008 bytes. Resume checkpoint, config, source snapshot and learning curve are saved. Full Kenai validation completed at 2026-10-03 14:35:29 UTC: AP50 **20.7500%**, AP50:95 **5.1727%**, AP75 **0.7800%**. Precision/recall at score 0.5 are59.5308%/17.3737%, with 78 false positives on20,529 negative frames and 1,276,003 saved detections. Evaluation charged 0.79627167 GPU-process hours. The motion screen fails against both published and paired-control references.

## Frozen final evaluation status

Seven matched seed-7 detectors at 672 are frozen in [freeze.json](artifacts/freeze.json), SHA-256 `141a160155504a796a4544f2461e418851d87a1d31e0ea476dc66a77cf907f71`. Every model completed one full source-only Channel evaluation, with unchanged code, preprocessing and thresholds. Prior annotation-header exposure remains disclosed. No new recipe passes the Kenai replication rule, and none has a replicated benefit claim. Original A448 three-seed evidence remains negative. The original full fraction/location study is incomplete.

| Model | Channel AP50 (%) | Channel AP50:95 (%) | Negative-frame FP / 1,250 frames |
|---|---:|---:|---:|
| Published frozen | 12.8548 | 3.7803 | 143 |
| A frozen | 10.9576 | 2.9263 | 6 |
| B frozen | 9.3323 | 2.5926 | 4 |
| B-CONTROL frozen | 9.5221 | 2.3977 | 0 |
| C-MOTION frozen | 9.2027 | 2.3224 | 0 |
| Published full fine-tuning | 24.2848 | 7.3644 | 2,152 |
| B full fine-tuning | 21.7517 | 6.4665 | 2,359 |

All metrics, original-coordinate predictions and complete evaluator outputs are linked in RESULTS. The [final artifact check](artifacts/final_execution_verification.json) verifies all frozen identities and the unchanged historical ledger prefix. Published-frozen CPU rescoring reproduced metrics, arrays, parameters and per-image output; the initial strict stdout comparison failed only on runtime lines, and that failure is retained. This is author verification.

The follow-up charged 17.96083750 of 24 GPU-process hours; Channel used 2.88519500 of its six-hour reserve. Overall recorded spend is 29.46504664 hours, leaving 70.53495336 from the owner-stated balance. The strongest AP model has substantial target false positives and has not been validated for operational use. Lower false-positive counts in weak frozen models must be interpreted alongside their low recall.

## Localization follow-up — completed author execution

The seed-7 image branch passes the practical screen. B initialization provides no label-efficiency improvement in the completed matched pilot. These artifacts are single-frame sonar fish detectors,
trained with complete Kenai clip labels, not new foundation models. Published/DINOv2 source identity and
all prior adapted encoder artifacts remain as documented above. The new branch is additional supervised
processing; no target-site imagery or annotations are used.

Architecture: ViT-S/14 with four registers; a 672-pixel image input produces a 48×48 patch grid,
excluding CLS/register tokens. The original pyramid resamples one final-layer, 384-channel grid to
84/42/21/10 maps. Its 96-channel projections feed the Faster R-CNN RPN and ROIAlign. Each projection
is Conv1×1(384→96), GroupNorm(8)/GELU, Conv3×3(96→96). These levels share one backbone grid.
The published-reference detector has 28,501,145 parameters, including 22,058,112 in the encoder;
the image variant has 28,575,737. All are configured for supervised fine-tuning in this block.
Anchors retain sizes 16/32/64/128 and aspect ratios 0.5/1/2; train RPN pre/post-NMS limits are
1,000/500 and inference limits are 500/200. ROIAlign uses 7×7 output and sampling ratio 2.
Final score 0.001, NMS 0.5 and 100 detections/frame remain fixed. At the last 10×10 map, Torchvision
uses integer RPN stride 67 and nearest-power-of-two ROI scale 1/64. Both arms use this tested rounding.

The image branch undoes channel-0 ImageNet normalization, maps gray to [-1,1], preserves black padding,
and applies three Conv3×3/stride-2/GroupNorm(8)/GELU blocks, with channels 1→32→64→96.
It has 74,592 parameters. The stride-4 tensor is an intermediate activation; fusion uses the final
stride-8 tensor in P0 and its average-pool-2 output in P1, both at coefficient 0.1. P2/P3, anchors,
ROI scales and one-frame inference stay fixed. The declared capacity control has 77,376 parameters
and uses P0 as its input. It is approximately parameter-matched, with a different FLOP count.
Branch initialization preserves shared CPU/CUDA RNG. Added convolution-only MACs at 672 are
0.943 billion for image and 0.542 billion for capacity. Measured fit and inference timing are reported
separately and include shared-machine variation.

New final weights and predictions:

- [det-finetune-r672-f010-s7-mb4 weights](artifacts/det-finetune-r672-f010-s7-mb4/checkpoint.pt), SHA-256 `145fdf740c17c4e677849d86cec560b8e2a2dfe425eaa75274926984eaef7068`; [raw detections](artifacts/det-finetune-r672-f010-s7-mb4/val/predictions.json); [complete metrics](artifacts/det-finetune-r672-f010-s7-mb4/val/metrics.json); train/validation charges 1.207222/1.174588h.
- [det-image-finetune-r672-f010-s7-mb4 weights](artifacts/det-image-finetune-r672-f010-s7-mb4/checkpoint.pt), SHA-256 `f913331e608412e8a505c07abc3dd8cdbec8333f4fb4e2e091c13d40242d9998`; [raw detections](artifacts/det-image-finetune-r672-f010-s7-mb4/val/predictions.json); [complete metrics](artifacts/det-image-finetune-r672-f010-s7-mb4/val/metrics.json); train/validation charges 1.107574/1.332956h.
- [det-capacity-finetune-r672-f010-s7-mb4 weights](artifacts/det-capacity-finetune-r672-f010-s7-mb4/checkpoint.pt), SHA-256 `fc4e1628b0927d2e6d325db1d558e2a681d3d39bdd6d8b5503d9eee87c9a84ba`; [raw detections](artifacts/det-capacity-finetune-r672-f010-s7-mb4/val/predictions.json); [complete metrics](artifacts/det-capacity-finetune-r672-f010-s7-mb4/val/metrics.json); train/validation charges 0.934931/1.322956h.
- [det-b-finetune-r672-f010-s7-mb4 weights](artifacts/det-b-finetune-r672-f010-s7-mb4/checkpoint.pt), SHA-256 `a0d6d3ee789e8cbb6ea44cd8beddcedda311131f83032ef2648caca715a28291`; [raw detections](artifacts/det-b-finetune-r672-f010-s7-mb4/val/predictions.json); [complete metrics](artifacts/det-b-finetune-r672-f010-s7-mb4/val/metrics.json); train/validation charges 1.118498/1.302986h.
- [det-finetune-r672-f001-s7-mb4 weights](artifacts/det-finetune-r672-f001-s7-mb4/checkpoint.pt), SHA-256 `e36df12c2db5c7b57679297716c43b7ea811f371a17283212da3d8fb86387a7a`; [raw detections](artifacts/det-finetune-r672-f001-s7-mb4/val/predictions.json); [complete metrics](artifacts/det-finetune-r672-f001-s7-mb4/val/metrics.json); train/validation charges 1.063815/1.044045h.
- [det-b-finetune-r672-f001-s7-mb4 weights](artifacts/det-b-finetune-r672-f001-s7-mb4/checkpoint.pt), SHA-256 `0f3c35b1d16c8edba3ae2ecba4c210a7b8ff109cb5d8b282abb2255c8ac41bb1`; [raw detections](artifacts/det-b-finetune-r672-f001-s7-mb4/val/predictions.json); [complete metrics](artifacts/det-b-finetune-r672-f001-s7-mb4/val/metrics.json); train/validation charges 0.926580/1.224649h.

All new checkpoint files contain full model/optimizer/RNG/exposure resume state. The final verification
confirms finite weights,2,000 successful updates,16,000 presentations, exact within-seed/fraction exposures,
archived training-source identities, saved full AP and canonical evaluator replay. Fresh heads, label
streams and optimization budgets are matched within each pair. Published/B comparisons deliberately
differ in encoder initialization; reference/image/capacity arms start from the same published encoder.
The fresh microbatch 4 reference is the
valid comparator; old microbatch 8 scores are context rather than an identical numerical trajectory.

New full Kenai metrics use evaluator v2. Original Channel results remain historical v1 with the annotation-ID
sentinel limitation and prior exposure; transfer performance of any new image/capacity head is unmeasured.
No deployment suitability follows. Negative frames, small fish misses, false positives and source/target
shift remain material limitations. Replication and optional fraction omissions are explicit in RESULTS.
No independent reviewer has certified these artifacts.

New block: **13.91030417/24 GPU-process hours**; historical cumulative baseline29.46504664h is retained. Overall spend **43.37535080/100h**, remaining **56.62464920h**. Original and previous block caps are not reset. The complete append-only ledger includes profiles, evaluation and the failed batch8 fit.
