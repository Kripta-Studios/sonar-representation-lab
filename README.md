# sonar-representation-lab

New CFC v1.1 grayscale fish detection study of DINOv2 self-supervised sonar adaptation. Author-evaluated exploratory research; see PROTOCOL.md and RESULTS.md for claims and execution status.

Windows private environment:

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe pycocotools psutil pytest ruff matplotlib
.venv/Scripts/python.exe src/acquire.py metadata --extract
.venv/Scripts/python.exe src/acquire.py kenai --extract
```

Data location is D:/sonar-representation-lab-data because E: cannot hold the archive plus extraction/cache. Acquisition downloads only named files, verifies MD5, records SHA-256, checks available space, and follows fresh publisher redirects. Channel download is reserved for the frozen final evaluation.

Pinned official backbone source (inspected before execution):

```powershell
git clone https://github.com/facebookresearch/dinov2.git vendor/dinov2
git -C vendor/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
New-Item -ItemType Directory -Force -Path artifacts/pretrained | Out-Null
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth -OutFile artifacts/pretrained/dinov2_vits14_reg4_pretrain.pth
```

Official checkpoint: [dinov2_vits14_reg4_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth), saved as `artifacts/pretrained/dinov2_vits14_reg4_pretrain.pth`. SHA-256: `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`. The factory validates source revision and checkpoint identity. No remote installation script is used. `requirements-lock.txt` records the actual private environment; reinstall with `uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128`.

Straightforward commands (research results require completed training, not just invocation):

```powershell
# Interrupted large transfers can use fresh publisher HTTP ranges.
.venv/Scripts/python.exe src/acquire.py kenai --ranges --extract
.venv/Scripts/python.exe src/data.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check src tests

# First milestone, serially: published frozen detector, full validation, SSL A, matched adapted detector.
.venv/Scripts/python.exe src/run_block.py --stage milestone
# Primary paired seed replicates and predeclared seed-7 fine-tuned/random controls.
.venv/Scripts/python.exe src/run_block.py --stage replicate
# Conditional 1%/100% paired extensions, reserving hours for held-out evaluation.
.venv/Scripts/python.exe src/run_block.py --stage fractions

# Equivalent individual commands for seed 7, 10%:
.venv/Scripts/python.exe src/train_detector.py train --kind published --fraction 10 --seed 7 --batch 8 --accumulation 1 --steps 2000 --out artifacts/det-published-f010-s7
.venv/Scripts/python.exe src/train_detector.py predict --checkpoint artifacts/det-published-f010-s7/checkpoint.pt --out artifacts/det-published-f010-s7/val
.venv/Scripts/python.exe src/adapt.py --seed 7 --batch 16 --accumulation 1 --steps 2000 --out artifacts/ssl-A-s7
.venv/Scripts/python.exe src/train_detector.py train --kind adapted --adapted artifacts/ssl-A-s7/encoder.pt --fraction 10 --seed 7 --batch 8 --accumulation 1 --steps 2000 --out artifacts/det-adapted-f010-s7
.venv/Scripts/python.exe src/train_detector.py predict --checkpoint artifacts/det-adapted-f010-s7/checkpoint.pt --out artifacts/det-adapted-f010-s7/val

# Recompute metrics from saved predictions without training or image access.
.venv/Scripts/python.exe src/evaluate.py --annotations D:/sonar-representation-lab-data/manifests/val.json --predictions artifacts/det-published-f010-s7/val/predictions.json --out artifacts/rescored-published-s7
```

Both trainers support `--resume <checkpoint.pt>` with the same configuration. An interrupted detector run must complete its declared update budget before final prediction. `--stop-after` supports a clean checkpoint/resume check; `adapt.py --profile` is explicitly a resource probe and never exports a research encoder. Use `src/profile_batch.py --batch 8` for real TRAIN batch/geometry/reload/resume verification.

Outputs: per-run `config.json`, `checkpoint.pt`, `curve.jsonl`, `exposure_summary.json`, `resources.json`, and console logs; adapted encoders in `ssl-A-s*/encoder.pt`; validation predictions and complete COCO evaluator output under each detector's `val/`. `artifacts/resource_ledger.jsonl` charges profiling, training, inference and failures to the initial block and overall balance. The resource guard accounts for physical WDDM GPU usage and owns a single-process lock; it never stops unrelated processes.

The initial 24 GPU-hour cap follows the user's report of 100 hours remaining overall. This project never accesses the stopped AEON work or creates a reviewer/signature prerequisite. The project-local Python platform fallback handles a host WMI identity-query error without changing Windows services or permissions.

Once the complete comparison is fixed, held-out evaluation and report generation use:

```powershell
.venv/Scripts/python.exe src/freeze.py
.venv/Scripts/python.exe src/channel.py
.venv/Scripts/python.exe src/summarize.py --errors
```

The freeze file records model/code hashes and preprocessing; it is an author experiment record, not independent review or an LLM approval. The Channel command downloads only the official grayscale archive and performs one fixed evaluation block. Any exposure or failure is preserved; the command refuses silent repeats. `artifacts/results.csv` contains every completed seed, `results_summary.json` gives means/sample standard deviations, and error-bin JSON files describe validation recall by resized fish short-side size. PNG learning curves are drawn from actual logged updates.
