# Model artifacts — sonar-representation-lab

Author-evaluated exploratory research. No independent review has occurred. This card was created after a real trained detector checkpoint existed, on 2026-10-01. Final metrics and completion status belong in [RESULTS.md](RESULTS.md).

## Current artifact status

The published-backbone detector has completed 2,000 real updates and saved final weights at `artifacts/det-published-f010-s7/checkpoint.pt`. Its final checkpoint successfully reloaded and was evaluated on all 30,454 Kenai validation frames: AP50 15.2853%, AP50:95 4.0586%. Complete predictions and evaluator output are in its `val/` directory. The checkpoint contains the model, optimizer, completed-update count, configuration, random-number states and image-exposure counts. The prediction command rejects an incomplete declared budget.

Genuine SSL configuration A / seed 7 completed 2,000 updates on 2026-10-02 at 11:43:51 UTC and exported the final EMA teacher encoder at `artifacts/ssl-A-s7/encoder.pt` (88,297,055 bytes; SHA-256 `15b3356ef61f163a955cefd240b04555635b104f7b222dcd408fdb783eaca210`). It processed 32,000 TRAIN frame presentations across 29,017 unique images from 162,198 available frames. Relative L2 change from the published initialization is 0.00298608; weight change demonstrates executed adaptation, not improved detection. The fresh matched 10% seed-7 detector completed all 2,000 updates at 2026-10-02 12:22:07 UTC. Its checkpoint is `artifacts/det-adapted-f010-s7/checkpoint.pt`, SHA-256 `346ca4ca64a64bf05058145b0458cc4156aa476139ec3995c7a2b3e938fb46aa`. CPU reload verified finite detector tensors, head optimizer counters at 2,000, all 176 encoder tensors bitwise frozen, and frame-exposure counters exactly equal to the baseline (16,000 presentations / 10,356 unique images). Full validation completed on all 30,454 frames: AP50 10.6641%, AP50:95 2.4921%, precision 42.3046%, recall 5.7194% at score 0.5 / IoU 0.5. Negative-frame false positives: 25 across 20,529 negative frames. The seed-7 AP50 difference is -4.6212 percentage points versus the matched published frozen baseline. This first seed does not establish a replicated effect. Complete predictions and evaluator output are retained in its `val/` directory. Earlier SSL profile artifacts are engineering probes and do not serve as adapted research encoders.

## Architecture and inputs

Official DINOv2 ViT-S/14 with four registers supplies a spatial patch grid to a common four-level pyramid and Faster R-CNN detector. The main comparison freezes the encoder and trains fresh, matched detection heads. Supervised fine-tuning and random frozen features are separately named controls.

Inputs are CFC v1.1 grayscale sonar frames, aspect-preserving resized into a 448-by-448 canvas with top-left black padding. The grayscale plane is repeated into three channels and ImageNet normalization is applied. Repetition does not represent three acoustic frequencies. Saved boxes use original-image coordinates, COCO image/category identities and xywh format.

## Initialization and data

Official source revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Published checkpoint: `dinov2_vits14_reg4_pretrain.pth`; SHA-256 `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`.

The first detector uses the fixed nested 10% Kenai TRAIN subset: 49 clips, 16,859 frames, 11,728 boxes and 10,705 negative frames. Every official annotation in the selected clips remains available. Train/validation membership follows complete official clip-level lists. The declared SSL run reads only the separate Kenai TRAIN image manifest; no boxes, classes, tracks, validation images or Channel imagery enter SSL.

See [PROTOCOL.md](PROTOCOL.md) for optimizer budgets, augmentations, selection policy, source links, dataset terms and the recorded Channel metadata-header exposure. The source DINOv2 repository uses Apache-2.0; the downloaded CFC record declares MIT rights and its publisher terms were saved locally.

## Intended use and limits

These artifacts support local research on reusable sonar features and fish bounding-box detection. They are not validated for operational deployment, species identification, biomass, tracking or counting. Kenai validation and one frozen Channel evaluation can support only the corresponding measured claims, not a full CFC benchmark or SOTA claim. Missing training is not a negative adaptation result.

Exact commands are in [README.md](README.md). Per-run configuration, source hashes, exposure counts, learning curves and resource records accompany each checkpoint. Retain these records when reusing an artifact; compare completed models using the same preprocessing and evaluator.
