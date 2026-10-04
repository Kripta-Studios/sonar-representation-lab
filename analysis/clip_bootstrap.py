"""Paired whole-clip bootstrap of pooled COCO AP from saved image matching."""

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from acquire import ROOT, save_json  # noqa: E402


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def stream_rows(path):
    """Decode the saved JSON array one row at a time, without dataset-sized objects."""
    decoder, buffer, ended = json.JSONDecoder(), "", False
    with gzip.open(path, "rt", encoding="utf8") as stream:
        while True:
            buffer = buffer.lstrip(" \t\r\n,[")
            if buffer.startswith("]"):
                return
            if not buffer and ended:
                raise ValueError("Incomplete evaluator JSON array")
            try:
                row, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(65536)
                if not chunk:
                    if ended:
                        raise
                    ended = True
                buffer += chunk
                continue
            buffer = buffer[offset:]
            yield row


def remap_clips(gt, predictions, selected):
    """Explicit known-answer reference: duplicate clips get fresh image/annotation IDs."""
    clips = {}
    for image in sorted(gt["images"], key=lambda item: item["id"]):
        clips.setdefault(Path(image["file_name"]).stem.rsplit("_", 1)[0], []).append(image)
    annotations, detections = {}, {}
    for row in gt["annotations"]:
        annotations.setdefault(row["image_id"], []).append(row)
    for row in predictions:
        detections.setdefault(row["image_id"], []).append(row)
    truth = {"images": [], "annotations": [], "categories": gt["categories"], "info": {}}
    output = []
    for clip in selected:
        for image in clips[clip]:
            image_id = len(truth["images"])
            truth["images"].append({**image, "id": image_id})
            for row in annotations.get(image["id"], []):
                truth["annotations"].append(
                    {**row, "id": len(truth["annotations"]) + 1, "image_id": image_id}
                )
            output.extend({**row, "image_id": image_id} for row in detections.get(image["id"], []))
    return truth, output


def compact_rows(rows, images):
    """Keep all-area/max100 matches; grouping remains at whole-image/clip level."""
    by_id = {}
    zero_id_rows = []
    for row in rows:
        if row is None or row["category_id"] != 1 or row["aRng"] != [0, 10000000000.0]:
            continue
        size = min(100, len(row["dtScores"]))
        legacy_matches = np.asarray(row["dtMatches"], dtype=np.int64)[:, :size] > 0
        corrected_matches = legacy_matches.copy()
        if 0 in row["gtIds"]:
            zero_id_rows.append(row["image_id"])
            zero_index = row["gtIds"].index(0)
            lookup = {value: index for index, value in enumerate(row["dtIds"][:size])}
            for threshold, matches in enumerate(row["gtMatches"]):
                detection_id = matches[zero_index]
                if detection_id > 0:
                    corrected_matches[threshold, lookup[detection_id]] = True
        by_id[row["image_id"]] = {
            "scores": np.asarray(row["dtScores"][:size]),
            "matched": corrected_matches,
            "legacy_matched": legacy_matches,
            "ignored": np.asarray(row["dtIgnore"], dtype=bool)[:, :size],
            "gt": int(np.count_nonzero(np.logical_not(row["gtIgnore"]))),
        }
    scores, matched, legacy_matched, ignored, clips = [], [], [], [], {}
    offset = 0
    for image in sorted(images, key=lambda item: item["id"]):
        clip = Path(image["file_name"]).stem.rsplit("_", 1)[0]
        entry = clips.setdefault(clip, {"indices": [], "gt": 0})
        row = by_id.get(image["id"])
        if row is None:
            continue
        size = len(row["scores"])
        entry["indices"].extend(range(offset, offset + size))
        entry["gt"] += row["gt"]
        offset += size
        scores.extend(row["scores"])
        matched.append(row["matched"])
        legacy_matched.append(row["legacy_matched"])
        ignored.append(row["ignored"])
    for entry in clips.values():
        entry["indices"] = np.asarray(entry["indices"], dtype=np.int64)
    return {
        "scores": np.asarray(scores),
        "matched": np.concatenate(matched, axis=1),
        "legacy_matched": np.concatenate(legacy_matched, axis=1),
        "legacy_zero_annotation_image_ids": zero_id_rows,
        "ignored": np.concatenate(ignored, axis=1),
        "clips": clips,
    }


def pooled_ap(compact, selected, original_order=False):
    # Concatenation follows fresh sequential virtual image IDs for each clip draw.
    # Matching cannot cross image boundaries; only match booleans survive reduction.
    indices = np.concatenate([compact["clips"][clip]["indices"] for clip in selected])
    if original_order:
        if len(set(selected)) != len(selected):
            raise ValueError("Original image-ID order is only a nonduplicated full-set reference")
        indices.sort()
    count = sum(compact["clips"][clip]["gt"] for clip in selected)
    if not count:
        return np.full(10, np.nan)
    indices = indices[np.argsort(-compact["scores"][indices], kind="mergesort")]
    found = compact["matched"][:, indices]
    valid = ~compact["ignored"][:, indices]
    tp = np.cumsum(found & valid, axis=1, dtype=np.float64)
    fp = np.cumsum(~found & valid, axis=1, dtype=np.float64)
    values = []
    for true, false in zip(tp, fp):
        recall = true / count
        precision = true / (true + false + np.spacing(1))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        lookup = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
        result = np.zeros(101)
        available = lookup < len(precision)
        result[available] = precision[lookup[available]]
        values.append(float(result.mean()))
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("artifacts/localization-block/saved_analysis"))
    args = parser.parse_args()
    if not 1 <= args.draws <= 200:
        raise ValueError("Bounded bootstrap requires1..200 draws")
    if (args.out / "bootstrap.json").exists():
        raise ValueError("Preserve completed analysis")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    gt = json.loads((ROOT / "manifests/val.json").read_text())
    clips = sorted({Path(image["file_name"]).stem.rsplit("_", 1)[0] for image in gt["images"]})
    rng = np.random.default_rng(7003)
    draws = rng.integers(0, len(clips), size=(args.draws, len(clips)))
    save_json(
        args.out / "draws.json",
        {
            "seed": 7003,
            "clips": clips,
            "draw_indices": draws.tolist(),
            "remapping": "Each draw occurrence has fresh sequential image IDs in clip-draw/image-ID order; annotations fresh sequential IDs; cache retains only image-local match booleans.",
        },
    )
    names = [
        "det-finetune-r672-f010-s7",
        "det-b-finetune-r672-f010-s7",
        "det-control-r672-f010-s7",
        "det-motion-r672-f010-s7",
    ]
    outputs, identities, operating = {}, {}, {}
    for name in names:
        path = PROJECT / "artifacts" / name / "val"
        compact = compact_rows(stream_rows(path / "coco_eval_images.json.gz"), gt["images"])
        reference = pooled_ap({**compact, "matched": compact["legacy_matched"]}, clips, original_order=True)
        corrected_reference = pooled_ap(compact, clips, original_order=True)
        metrics = json.loads((path / "metrics.json").read_text())
        save_json(
            args.out / f"{name}-full-reference.json",
            {
                "computed_AP50": float(reference[0]),
                "saved_AP50": metrics["AP50"],
                "computed_AP95": float(reference.mean()),
                "saved_AP95": metrics["AP50:95"],
                "order": "original sorted image IDs, including stable score-tie order",
                "positive_id_AP50": float(corrected_reference[0]),
                "positive_id_AP95": float(corrected_reference.mean()),
                "legacy_zero_annotation_image_ids": compact["legacy_zero_annotation_image_ids"],
                "bootstrap_evaluator": "positive annotation IDs; zero-ID match recovered via gtMatches/dtIds",
            },
        )
        assert abs(reference[0] - metrics["AP50"]) < 1e-12
        assert abs(reference.mean() - metrics["AP50:95"]) < 1e-12
        values = []
        for index, draw in enumerate(draws):
            ap = pooled_ap(compact, [clips[i] for i in draw])
            values.append([float(ap[0]), float(ap.mean()), float(ap[5])])
            if index % 20 == 0:
                memory = psutil.Process().memory_info()
                if max(memory.rss, getattr(memory, "peak_wset", memory.rss)) >= 22 * 1024**3:
                    raise RuntimeError("Owned RAM cap reached")
                print(
                    json.dumps(
                        {
                            "run": name,
                            "draw": index,
                            "cpu_seconds": time.monotonic() - started,
                            "rss_bytes": memory.rss,
                        }
                    ),
                    flush=True,
                )
        outputs[name] = values
        identities[name] = {
            "per_image_coco_sha256": file_sha(path / "coco_eval_images.json.gz"),
            "full_ap_exact": True,
            "positive_id_AP50": float(corrected_reference[0]),
            "positive_id_AP95": float(corrected_reference.mean()),
            "positive_id_AP75": float(corrected_reference[5]),
        }
        diagnostics = json.loads((PROJECT / "artifacts/kenai_diagnostics" / f"{name}.json").read_text())
        eligible = [
            point
            for point in diagnostics["operating_points"]
            if point["false_positives_per_negative_frame"] <= 0.05
        ]
        best = max(eligible, key=lambda point: point["recall"]) if eligible else None
        operating[name] = {
            "FP_per_negative_budget": 0.05,
            "point": best,
            "status": "Kenai exploratory grid selection; no Channel use",
        }
        save_json(
            args.out / f"{name}.json",
            {
                "bootstrap_values_AP50_AP95_AP75": values,
                "identity": identities[name],
                "operating": operating[name],
            },
        )
        del compact
    contrasts = []
    for baseline, candidate in ((names[0], names[1]), (names[2], names[3])):
        delta = 100 * (np.asarray(outputs[candidate]) - np.asarray(outputs[baseline]))
        contrasts.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "metrics": ["AP50", "AP50:95", "AP75"],
                "paired_delta_percentile95_pp": np.nanpercentile(delta, [2.5, 97.5], axis=0).tolist(),
                "paired_delta_median_pp": np.nanmedian(delta, axis=0).tolist(),
            }
        )
    save_json(
        args.out / "bootstrap.json",
        {
            "contrasts": contrasts,
            "draws": args.draws,
            "seed": 7003,
            "identities": identities,
            "cpu_wall_seconds": time.monotonic() - started,
            "owned_native_peak_bytes": getattr(
                psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss
            ),
            "status": "author CPU analysis; conditional clip-sampling uncertainty, not optimizer-seed/location replication",
            "gpu_used": False,
        },
    )


if __name__ == "__main__":
    main()
