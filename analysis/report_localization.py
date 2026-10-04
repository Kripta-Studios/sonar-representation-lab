"""Reconcile the existing four core documents with completed localization artifacts."""

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from acquire import save_json  # noqa: E402
from budget import allocation_limits  # noqa: E402
from runtime import LEDGER, file_sha  # noqa: E402

BLOCK = PROJECT / "artifacts/localization-block"
MARKER = "## Localization follow-up — completed author execution"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def append(path, text):
    old = path.read_text(encoding="utf8")
    if MARKER in old:
        raise ValueError(f"Preserve existing reconciliation: {path}")
    path.write_text(old.rstrip() + "\n\n" + MARKER + "\n\n" + text.rstrip() + "\n", encoding="utf8")


def protocol():
    closed = read(BLOCK / "development_closed.json")
    roster = "\n".join(f"- `{name}`" for name in closed["roster"])
    omissions = "\n".join(f"- {name}: {reason}" for name, reason in closed["omissions"].items())
    append(
        PROJECT / "PROTOCOL.md",
        f"""This new finite development block is closed. Its exact decision record is
`artifacts/localization-block/development_closed.json`. The screen and all poor/failed attempts remain preserved.
The selected numerical protocol is microbatch 4/accumulation 2, effective batch 8, 2,000 successful updates,
16,000 presentations, 672 input, complete nested clip labels, unchanged optimizer/loss/anchor/NMS settings.
The earlier microbatch 1 text describes a tested fallback; the selected arms all use microbatch 4.

The frozen final Kenai roster is:

{roster}

Omitted cells:

{omissions or "- None of the conditionally admitted cells was omitted."}
- Original full 100%-label grid: DEFERRED_BY_OWNER_PRIORITY_AMENDMENT.

Two concrete correctness findings were repaired before any completed new image-branch score. First,
the publisher's valid annotation ID 0 collides with COCO's unmatched-zero sentinel. Evaluatorv2 deep-copies
annotations and remaps IDs to positive integers without changing source files, image/category IDs or boxes.
New predictions use the canonical v2 evaluator. The historical 15-run Kenai correction audit uses saved
matching, preserves v1 metrics, and reports selected corrected AP metrics separately; the largest AP50
change is+0.004833 percentage points. Historical Channel scores retain their v1 limitation. No Channel
inference, annotation inspection or outcome-based model repair occurs in this block.

Second, optional CPU branch construction formerly called torch.manual_seed inside a CPU-only fork,
which also reseeded CUDA. A dedicated CPU Generator now supplies only the forked CPU state. Both shared
CPU and CUDA RNG streams remain intact; tests passed. The interrupted batch8 one-update image attempt
predates this repair and remains incomplete. The completed fresh published reference uses no branch,
so its trajectory is unaffected. Its actual training_source_snapshot.zip matches every recorded training
source/protocol hash; the later source_snapshot.zip identifies reconstruction/inference code. Other new
fits retain their matching training source snapshots. These are versioned engineering corrections,
not outcome-guided candidate changes.

The microbatch 4 image fit also encountered Windows error 1455 in the OS child-process memory query.
Its failed interval charged0.4199305556h and retained971 successful updates/7,768 presentations.
Eight interrupted-update presentations were discarded; model/optimizer/RNG were exactly reloaded
from the immutable step971 checkpoint after measured GPU/system memory capacity recovered. Direct
CLI resume avoided keeping the supervisor's Torch memory resident. The source/config and scientific
settings were unchanged; no resource check, paging-file policy or unrelated process was altered.
Successful presentation counters are distinct from attempted/discarded computation. Future runtime
forecasts include every charged fit interval, including retries and save/resume phases, before
conditional admission; failed charges appear only once in the append-only overall ledger.

Final freeze: `artifacts/localization-block/final_freeze.json`, at most 12 models, checkpoint/prediction/
evaluator/source/input identities and preserved allocation. Reserve 6 GPU-process hours for final operations.
Verify all finite checkpoints, successful-update/exposure counts, matched within-seed/fraction streams,
saved full AP, then canonical CPU replay of the first published reference. Run bounded proposal inference
on the same recorded 512 Kenai frames for the primary pair and latency on eight fixed real TRAIN frames
at batch1/4. These descriptive checks do not select or repair a model. Final figure bars represent individual
checkpoints. No new Channel measurement or new SSL fitting is authorized by this localization block.
Documentation updates after execution cannot retrospectively create independent review or a prospective
hypothesis. Previously adopted development choices remain post-hoc choices informed by Kenai.
""",
    )


def documents():
    closed = read(BLOCK / "development_closed.json")
    frozen = read(BLOCK / "final_freeze.json")
    verified = read(BLOCK / "final_verification.json")
    assert verified["all_metric_fields_match"] and verified["precision_recall_scores_bitwise_equal"]
    assert frozen["protocol_sha256"] == file_sha(PROJECT / "PROTOCOL.md")
    charges = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    rows = []
    for item in frozen["models"]:
        directory = PROJECT / item["path"]
        metric = read(directory / "val/metrics.json")
        parameter = read(directory / "parameter_summary.json")
        rows.append(
            {
                **item,
                "path": directory.relative_to(PROJECT).as_posix(),
                "metrics": metric,
                "parameters": parameter,
                "train_hours": sum(
                    r.get("charged_gpu_hours", 0) for r in charges if r.get("name") == directory.name
                ),
                "validation_hours": read(directory / "val/resources.json")["charged_gpu_hours"],
            }
        )
    table = [
        "| Completed run | AP50 (%) | AP50:95 (%) | AP75 (%) | P / R at0.5 (%) | FP / negative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        table.append(
            f"| [{row['run']}]({row['path']}/checkpoint.pt) | {100 * m['AP50']:.4f} | "
            f"{100 * m['AP50:95']:.4f} | {100 * m['AP75']:.4f} | "
            f"{100 * m['precision']:.2f} / {100 * m['recall']:.2f} | {m['false_positives_per_negative_frame']:.5f} |"
        )
    screen = closed["screen"]
    error_table = [
        "| Seed-7 checkpoint | Recall <4 /4–8 /8–16 /≥16 (%) | Duplicate / localization / background FP |",
        "|---|---|---|",
    ]
    for row in (r for r in rows if r["seed"] == 7 and r["fraction"] == 10 and not r["adapted_sha256"]):
        diagnostic = read(BLOCK / "diagnostics" / f"{row['run']}.json")
        point = next(p for p in diagnostic["operating_points"] if p["score_threshold"] == 0.5)
        bins = point["size_bins"]
        recalls = " / ".join(f"{100 * bins[k]['recall']:.2f}" for k in ("<4px", "4-8px", "8-16px", ">=16px"))
        counts = point["false_positive_types"]
        fp_counts = " / ".join(str(counts[k]) for k in ("duplicate", "localization", "background"))
        error_table.append(f"| {row['run']} | {recalls} | {fp_counts} |")
    operating_table = [
        "| Historical model | Kenai selected score | Precision / recall (%) | FP / negative |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "det-finetune-r672-f010-s7",
        "det-b-finetune-r672-f010-s7",
        "det-control-r672-f010-s7",
        "det-motion-r672-f010-s7",
    ):
        point = read(BLOCK / "saved_analysis-v4" / f"{name}.json")["operating"]["point"]
        operating_table.append(
            f"| {name} | {point['score_threshold']} | {100 * point['precision']:.2f} / {100 * point['recall']:.2f} | {point['false_positives_per_negative_frame']:.5f} |"
        )
    capacity = next((r for r in rows if r["neck"] == "capacity"), None)
    capacity_note = "The capacity-control cell was omitted; the fine-scale-input mechanism remains untested."
    if capacity:
        capacity_note = (
            f"At seed 7, capacity minus the matched published reference changes AP50 by "
            f"{100 * (capacity['metrics']['AP50'] - rows[0]['metrics']['AP50']):+.4f} percentage points "
            f"and AP50:95 by {100 * (capacity['metrics']['AP50:95'] - rows[0]['metrics']['AP50:95']):+.4f}. "
            f"Image minus capacity changes AP50 by "
            f"{100 * (rows[1]['metrics']['AP50'] - capacity['metrics']['AP50']):+.4f} points "
            f"and AP50:95 by {100 * (rows[1]['metrics']['AP50:95'] - capacity['metrics']['AP50:95']):+.4f}. "
            "These are observed checkpoint contrasts, not replicated evidence isolating input detail from compute."
        )
    proposal_table = [
        "| Primary checkpoint | RPN recall@200 IoU0.5 /0.75 (%) | Covered but no final coverage | No RPN coverage |",
        "|---|---:|---:|---:|",
    ]
    for row in rows[:2]:
        probe = read(BLOCK / "final_proposals" / row["run"] / "metrics.json")
        assert probe["checkpoint_sha256"] == row["files"]["checkpoint.pt"]
        assert probe["frames"] == 512 and probe["first_batch_hook_predictions_bitwise_equal"]
        recall = probe["proposal_recall"]
        counts = probe["counts"]
        proposal_table.append(
            f"| {row['run']} | {100 * recall['covered_at_200_iou_0.5']:.2f} / {100 * recall['covered_at_200_iou_0.75']:.2f} | {counts['proposal_covered_but_no_final_coverage']} | {counts['no_proposal_coverage']} |"
        )
    latency = read(BLOCK / "final_latency/latency.json")
    latency_table = [
        "| Primary checkpoint | Batch1 warm ms/frame | Batch4 warm ms/frame |",
        "|---|---:|---:|",
    ]
    for row in latency["models"]:
        values = {v["batch"]: v["wall_median_ms_per_frame"] for v in row["measurements"]}
        latency_table.append(f"| {row['run']} | {values[1]:.3f} | {values[4]:.3f} |")
    if not screen["passes"]:
        next_decision = "Keep the matched published detector as the practical reference. This fixed image-branch intervention did not meet the allocation rule; do not add another SSL objective to rescue it. The next separately authorized experiment should test a matched longer detector optimization budget, with terminal updates and localization/negative-frame diagnostics fixed before fitting. The present 2,000-step screen cannot establish convergence or a fully trained ceiling."
    elif (
        capacity
        and capacity["metrics"]["AP50"] >= rows[1]["metrics"]["AP50"]
        and capacity["metrics"]["AP50:95"] >= rows[1]["metrics"]["AP50:95"]
    ):
        next_decision = "The capacity control is at least as strong on both AP metrics, so added detector processing remains a sufficient alternative explanation; a fine-scale-input mechanism is unestablished. Prioritize a separately authorized replicated capacity/image/reference contrast at the same training budget before any new SSL recipe or location claim."
    else:
        next_decision = "The image pilot passes the practical rule, but mechanism and transfer remain limited by the actual capacity-control and replication evidence. The next separately authorized experiment should complete any missing matched capacity/reference/image replications, then freeze a new source-only roster before a new-location test. No Channel repair or new SSL objective follows automatically from this detector gain."
    fraction_note = "The optional matched1%/10% initialization curve did not execute; no label-efficiency result is invented."
    one = [r for r in rows if r["fraction"] == 1 and r["neck"] == "none"]
    one_published = next((r for r in one if not r["adapted_sha256"]), None)
    one_b = next((r for r in one if r["adapted_sha256"]), None)
    if one_published and one_b:
        gain50 = 100 * (one_b["metrics"]["AP50"] - one_published["metrics"]["AP50"])
        gain95 = 100 * (one_b["metrics"]["AP50:95"] - one_published["metrics"]["AP50:95"])
        fraction_note = (
            f"At 1% labels, B minus published full fine-tuning changes AP50 by {gain50:+.4f}pp "
            f"and AP50:95 by {gain95:+.4f}pp. Both points use the same five complete clips, "
            "seed 7, microbatch 4/effective 8 and 2,000 updates. This is one fixed subset/optimizer-seed curve; "
            "it is not a replicated label-efficiency claim or frozen-feature benefit."
        )
        ten_b = next(
            (r for r in rows if r["fraction"] == 10 and r["neck"] == "none" and r["adapted_sha256"]),
            None,
        )
        if ten_b:
            fraction_note += (
                f" The freshly trained numerical bridge at 10% labels gives B minus published "
                f"AP50 {100 * (ten_b['metrics']['AP50'] - rows[0]['metrics']['AP50']):+.4f}pp and "
                f"AP50:95 {100 * (ten_b['metrics']['AP50:95'] - rows[0]['metrics']['AP50:95']):+.4f}pp. "
                "Both fractions use the same microbatch and fixed optimization budget; the old microbatch-8 "
                "B endpoint is historical context rather than the matched endpoint of this curve."
            )
        if gain50 >= 1 and gain95 >= 0:
            next_decision += (
                " The exploratory 1% B initialization contrast shows a practical source-site gain; "
                "replicating that matched full-fine-tuning contrast with all seeds is also a priority "
                "before attributing label efficiency to SSL. This does not reopen development or admit another SSL configuration."
            )
        elif gain50 < 0 and gain95 < 0:
            next_decision += (
                " B initialization loses both AP metrics at 1% as well as 10% in the matched seed-7 "
                "screen. Do not allocate the next block to further tuning of this recipe. Retain these "
                "negative results and prioritize replicated supervised localization comparisons."
            )
    conclusion = (
        "The seed-7 image branch passes the practical screen."
        if screen["passes"]
        else "The seed-7 image branch fails the practical screen; no qualifying gain is established."
    )
    if one_published and one_b and one_b["metrics"]["AP50"] < one_published["metrics"]["AP50"]:
        conclusion += (
            " B initialization provides no label-efficiency improvement in the completed matched pilot."
        )
    paired = []
    for seed in (7, 13, 23):
        group = [r for r in rows if r["seed"] == seed and r["fraction"] == 10 and not r["adapted_sha256"]]
        reference = next((r for r in group if r["neck"] == "none"), None)
        image = next((r for r in group if r["neck"] == "image"), None)
        if reference and image:
            paired.append(
                {
                    "seed": seed,
                    "AP50_delta_pp": 100 * (image["metrics"]["AP50"] - reference["metrics"]["AP50"]),
                    "AP95_delta_pp": 100 * (image["metrics"]["AP50:95"] - reference["metrics"]["AP50:95"]),
                }
            )
    replicate = "\n".join(
        f"- Seed{r['seed']}: image minus published AP50 {r['AP50_delta_pp']:+.4f}pp; AP50:95 {r['AP95_delta_pp']:+.4f}pp."
        for r in paired
    )
    if len(paired) == 3:
        replicate += f"\n- Mean paired AP50 change {statistics.mean(r['AP50_delta_pp'] for r in paired):+.4f}pp; sample SD {statistics.stdev(r['AP50_delta_pp'] for r in paired):.4f}pp. Every seed is included."
        replicate += f"\n- Mean paired AP50:95 change {statistics.mean(r['AP95_delta_pp'] for r in paired):+.4f}pp; sample SD {statistics.stdev(r['AP95_delta_pp'] for r in paired):.4f}pp."
    else:
        replicate += "\n- This contrast remains a single training-seed pilot, not a replicated benefit claim."
    limits = allocation_limits(LEDGER, BLOCK / "allocation.json", "final", BLOCK / "final_freeze.json")
    total = sum(r.get("charged_gpu_hours", 0) for r in charges)
    spend = total - limits["baseline_gpu_hours"]
    resources = f"New block: **{spend:.8f}/24 GPU-process hours**; historical cumulative baseline29.46504664h is retained. Overall spend **{total:.8f}/100h**, remaining **{100 - total:.8f}h**. Original and previous block caps are not reset. The complete append-only ledger includes profiles, evaluation and the failed batch8 fit."
    omissions = "\n".join(f"- {k}: {v}." for k, v in closed["omissions"].items())
    artifacts = "\n".join(
        f"- [{r['run']} weights]({r['path']}/checkpoint.pt), SHA-256 `{r['files']['checkpoint.pt']}`; [raw detections]({r['path']}/val/predictions.json); [complete metrics]({r['path']}/val/metrics.json); train/validation charges {r['train_hours']:.6f}/{r['validation_hours']:.6f}h."
        for r in rows
    )
    cmds = """```powershell
# Exact finite driver invocation used; do not restart completed development.
$env:SONAR_RESEARCH_ALLOCATION='artifacts/localization-block/allocation.json'
$env:SONAR_RESEARCH_PHASE='development'
$env:SONAR_ORIGINAL_BLOCK_OPERATION='0'
.venv/Scripts/python.exe -u analysis/run_localization_block.py --microbatch 4 --profile artifacts/localization-block/profile-image-mb4/profile.json
# Completed saved-prediction analyses; existing outputs are protected against overwrite.
.venv/Scripts/python.exe -u analysis/clip_bootstrap.py --out artifacts/localization-block/saved_analysis-v4
.venv/Scripts/python.exe -u analysis/evaluator_correction_audit.py
.venv/Scripts/python.exe -u analysis/evaluator_correction_audit.py --runs artifacts/det-adapted-f010-s7 --out artifacts/localization-block/evaluator_correction_audit_A7.json
.venv/Scripts/python.exe analysis/report_localization.py protocol
.venv/Scripts/python.exe analysis/finalize_localization.py freeze
.venv/Scripts/python.exe analysis/finalize_localization.py verify
$env:SONAR_RESEARCH_ALLOCATION='artifacts/localization-block/allocation.json'
$env:SONAR_RESEARCH_PHASE='final'
$env:SONAR_FINAL_FREEZE='artifacts/localization-block/final_freeze.json'
$env:SONAR_ORIGINAL_BLOCK_OPERATION='0'
.venv/Scripts/python.exe analysis/proposals.py --checkpoints artifacts/det-finetune-r672-f010-s7-mb4/checkpoint.pt artifacts/det-image-finetune-r672-f010-s7-mb4/checkpoint.pt --out artifacts/localization-block/final_proposals --batch 4 --allocation artifacts/localization-block/allocation.json
.venv/Scripts/python.exe analysis/latency.py --runs artifacts/det-finetune-r672-f010-s7-mb4 artifacts/det-image-finetune-r672-f010-s7-mb4 --out artifacts/localization-block/final_latency --batches 1 4
.venv/Scripts/python.exe analysis/plot_localization.py
.venv/Scripts/python.exe analysis/report_localization.py documents
```

Every actual fit/prediction argument vector, forecast, allocation and source identity is saved in its
`admission-*.json`; its console and config preserve exact execution. For a future explicitly allocated
fresh reproduction, use `src/train_detector.py train --kind finetune --size 672 --neck image` (or `none`/`capacity`)
with separate fresh output paths, existing manifest, seed, `--steps 2000 --batch 4 --accumulation 2` and
the recorded allocation. Exact primary image training invocation used:

```powershell
$env:SONAR_RESEARCH_ALLOCATION='artifacts/localization-block/allocation.json'
$env:SONAR_RESEARCH_PHASE='development'
$env:SONAR_ORIGINAL_BLOCK_OPERATION='0'
.venv/Scripts/python.exe -u src/train_detector.py train --kind finetune --size 672 --neck image --fraction 10 --seed 7 --steps 2000 --batch 4 --accumulation 2 --out artifacts/det-image-finetune-r672-f010-s7-mb4 --allocation artifacts/localization-block/allocation.json
# Exact recovery invocation after the preserved step-971 resource interruption.
.venv/Scripts/python.exe -u src/train_detector.py train --kind finetune --size 672 --neck image --fraction 10 --seed 7 --steps 2000 --batch 4 --accumulation 2 --out artifacts/det-image-finetune-r672-f010-s7-mb4 --allocation artifacts/localization-block/allocation.json --resume artifacts/det-image-finetune-r672-f010-s7-mb4/checkpoint.pt
.venv/Scripts/python.exe -u src/train_detector.py predict --checkpoint artifacts/det-image-finetune-r672-f010-s7-mb4/checkpoint.pt --out artifacts/det-image-finetune-r672-f010-s7-mb4/val --batch 4 --allocation artifacts/localization-block/allocation.json
```

These preserve executed commands; use fresh paths for a new reproduction rather than overwriting completed runs.
`predict --checkpoint ... --out ... --batch 4 --allocation ...` saves original-coordinate predictions;
`src/evaluate.py --annotations D:/sonar-representation-lab-data/manifests/val.json --predictions ... --out ...`
scores saved predictions without GPU or inference. Use fresh evaluator output paths and identify v2.
"""
    append(
        PROJECT / "RESULTS.md",
        f"""{conclusion} The primary matched changes are AP50 **{screen["AP50_gain_pp"]:+.4f}pp** and
AP50:95 **{screen["AP95_gain_pp"]:+.4f}pp**; fixed 448-bin below 16 recall changes from
{100 * screen["reference_small_recall"]:.3f}% to{100 * screen["small_recall"]:.3f}% at score 0.5.
All new cells are full supervised fine-tuning at 672 with matched microbatch 4/effective batch 8. These new
scores use positive-annotation-ID evaluator v2; historical tables above remain v1 snapshots.

Label policy remains complete official clips:10% provides49 TRAIN clips,16,859 frames,11,728 annotations
and10,705 negative frames;1% provides5 clips,1,952 frames,1,782 annotations and1,016 negatives.
These are available counts, not claims of exhaustive training. Each completed fit samples16,000
presentations with replacement. Validation retains30,454 frames/18,551 boxes/20,529 negatives in64 clips.
No adjacent-frame random split, partial fish labeling or negative-frame removal was introduced.

{chr(10).join(table)}

![Completed localization comparison](artifacts/localization-block/comparison.png)

{replicate}

{fraction_note}

Fixed 448-scale size bins and score 0.5 error diagnosis for the seed-7 reference and added branches:

{chr(10).join(error_table)}

Bounded512-frame Kenai proposal diagnosis, identical recorded selection in both arms:

{chr(10).join(proposal_table)}

This is object coverage, not one-to-one AP or full-validation proposal recall. The saved rows include
recall@100/200, IoU0.5/0.75, objectness and negative-frame proposals. The instrumentation's first batch
reproduces uninstrumented detections bitwise. Neither these values nor AP75 replace the primary screen.

Measured inference timing on the same eight real TRAIN frames:

{chr(10).join(latency_table)}

These are warm model-only medians, with inputs already on GPU; they exclude decoding, loading and scoring.
Shared-machine timing is descriptive, not a deployment throughput guarantee or proof of a recipe-speed effect.

The hypothesis tested here is access to fine-scale grayscale cues through added detector processing.
It is not a new SSL recipe or proof that14-pixel patch tokens intrinsically erase all subpatch information.
The capacity control, when completed, uses the existing P0 grid and differs slightly in parameter count;
its score must inform mechanism attribution. Replicating image/reference without replicating the capacity
control does not establish a replicated fine-scale-specific mechanism.
{capacity_note}
Analytical added-branch cost at 672 is942,907,392 convolution MACs for image versus541,900,800 for
capacity (1.886/1.084 billion FLOPs under two-operations-per-MAC counting). These exclude bias,
normalization, activation, pooling, fusion and the common detector; see architecture_costs.json.
Thus the control is approximately parameter-matched, not FLOP-matched.

The saved-output200-draw whole-clip bootstrap preserves64 clip units, duplicate-ID remapping and
stable score-tie conventions. B minus published fine-tuning AP50 percentile interval is[-2.3868,-0.2217]pp;
its AP50:95 interval is[-0.7058,+0.2106]pp. C minus B-CONTROL AP50 interval is[-0.3577,+0.8482]pp,
AP50:95[-0.3117,+0.1373]pp. These are conditional clip-sampling intervals, not optimizer-seed or
location uncertainty. They do not rescue the original failed practical screens. Existing Kenai-only
threshold-grid operating diagnostics are in`artifacts/localization-block/saved_analysis-v4`; thresholds
are exploratory source-site choices, with no target tuning or replacement of the0.5 historical diagnostic.

{chr(10).join(operating_table)}

Correctness disclosure: publisher GTannotation ID 0 collided with COCO's zero sentinel. All15 historical
Kenai scores were audited using saved matches, including the earliest A run's verified identical CPU
replay. Published fine-tuning448 and672 AP50 change by+0.001250 and+0.004833pp respectively; other
historical AP50 values do not change. Original outputs are untouched. The selected historical v2 Kenai
AP metrics are derived from cached matching, not invented new complete evaluator outputs. The new
roster has canonical completev2 output and an exact complete scorer replay. Historical Channel scores
remain v1 with this newly discovered evaluator limitation; this block accesses no Channel data.

Omissions and boundaries:

{omissions or "- All conditionally admitted cells completed."}
- Original100%-label grid remains DEFERRED_BY_OWNER_PRIORITY_AMENDMENT; no fully trained label ceiling.
- Any completed1%/10% curve uses one fixed nested subset seed and equal2,000-update budgets; it cannot establish a universal label-efficiency curve.
- No new self-supervised encoder was fitted in this localization block. The six real prior SSL encoders remain preserved; any B-labelled new fine-tune starts from the existing B checkpoint. Supervised encoder changes are retained inside every new detector checkpoint.
- No new location result, independent review, SOTA, fish species/biomass, tracking or counting claim.

{resources}

The first batch8 image attempt stopped at one successful update after aggregate physical GPU use reached
12,311,330,816bytes due concurrent background allocation. The guard stopped the owned process; the
checkpoint, source, error and0.0169052778h charge remain. It is incomplete, not a negative model result.
The selected microbatch 4 image fit retained a separate Windows memory-query interruption at step971;
its0.4199305556h charge, immutable checkpoint and eight discarded presentations are preserved. Exact
model/optimizer/RNG reload passed before continuing the same2,000-update budget. This was a resource
failure, not a detection result. Successful exposure counts do not silently include discarded updates.
All admitted successful new arms use profiled microbatch 4 with per-microbatch checks, unchanged10GiB
GPU/22GiB ownedRAM caps and preserved unrelated processes. Tests verify branch geometry/gradients,
shared initialization, CPU/CUDA RNG preservation, cache/resolution conventions, known-answer scorer
including annotation ID 0, and loaded checkpoint model/optimizer/RNG. BF16 replay has declared numerical
tolerances; only loaded state and final scorer arrays are asserted bitwise identical.

Artifacts:

{artifacts}

The final freeze, verification, figure data, proposal/latency diagnostics and ledger live under the existing
artifact structure. No application, second repository or reviewer dependency was introduced.

Next evidence-based decision: {next_decision}

The practical bottlenecks remain small-object misses, weak tight-box overlap and background false
positives. AP75 and the fixed size-bin counts above quantify localization and miss rates; proposal
coverage further distinguishes objects absent from the proposal set from failures after proposal
generation. These checks motivate the next matched detector experiment. They do not diagnose
configuration A as temporal collapse, identify biological motion, or justify another unbounded SSL search.

{cmds}
""",
    )
    append(
        PROJECT / "README.md",
        f"""The completed follow-up is recorded in[RESULTS.md](RESULTS.md),
[PROTOCOL.md](PROTOCOL.md) and[MODEL_CARD.md](MODEL_CARD.md). {conclusion}
The primary AP50 change is{screen["AP50_gain_pp"]:+.4f}pp against a fresh matched published full-fine-tuning
reference. New predictions use evaluator v2; historical scoring limitations remain disclosed.

Use the existing private environment/data. Detector CLI now accepts`--neck none|image|capacity`;
`--neck image` adds grayscale detail features without changing the common RPN/ROI geometry. All new
fits use672, microbatch 4/accumulation 2,2,000 successful updates and full clip labels. Saved predictions
can be scored with`src/evaluate.py`; no retraining is needed. Exact invocations and configs are retained
beside every run. `analysis/finalize_localization.py` checks the finite roster and replays its scorer;
`analysis/report_localization.py` reconciles these existing documents. Completed outputs are immutable.

{resources}

No new Channel evaluation occurs here. The original seven-model transfer block is preserved and exposed,
with its v1 evaluator limitation disclosed. The original complete fraction/location study remains incomplete.
New detector results are author execution, not independent review or evidence that SSL improved reusable
features. Historical planning sections above describe earlier stages, not unexecuted promises to restart them.
""",
    )
    append(
        PROJECT / "MODEL_CARD.md",
        f"""{conclusion} These artifacts are single-frame sonar fish detectors,
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

{artifacts}

All new checkpoint files contain full model/optimizer/RNG/exposure resume state. The final verification
confirms finite weights,2,000 successful updates,16,000 presentations, exact within-seed/fraction exposures,
archived training-source identities, saved full AP and canonical evaluator replay. Fresh heads and published
encoder initialization are matched within each supervised pair. The fresh microbatch 4 reference is the
valid comparator; old microbatch 8 scores are context rather than an identical numerical trajectory.

New full Kenai metrics use evaluator v2. Original Channel results remain historical v1 with the annotation-ID
sentinel limitation and prior exposure; transfer performance of any new image/capacity head is unmeasured.
No deployment suitability follows. Negative frames, small fish misses, false positives and source/target
shift remain material limitations. Replication and optional fraction omissions are explicit in RESULTS.
No independent reviewer has certified these artifacts.

{resources}
""",
    )
    # Keep the completed earlier block intact while making the latest result easy to find.
    for name in ("README.md", "RESULTS.md", "MODEL_CARD.md"):
        path = PROJECT / name
        content = path.read_text(encoding="utf8")
        heading, remainder = content.split("\n", 1)
        notice = (
            f"\nLatest completed follow-up: {conclusion} Primary AP50 change "
            f"{screen['AP50_gain_pp']:+.4f} percentage points. See the "
            "[localization follow-up](#localization-follow-up--completed-author-execution) "
            "below for new checkpoints, evaluator v2, resource charges and omissions. "
            "Earlier dated sections retain their historical results.\n"
        )
        path.write_text(heading + "\n" + notice + remainder, encoding="utf8")
    save_json(
        BLOCK / "documentation_reconciliation.json",
        {
            "rows": rows,
            "paired_deltas": paired,
            "new_block_gpu_hours": spend,
            "overall_gpu_hours": total,
            "document_hashes": {
                name: file_sha(PROJECT / name)
                for name in ("README.md", "RESULTS.md", "PROTOCOL.md", "MODEL_CARD.md")
            },
            "freeze_sha256": file_sha(BLOCK / "final_freeze.json"),
            "author_evaluated": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("protocol", "documents"))
    args = parser.parse_args()
    protocol() if args.action == "protocol" else documents()
