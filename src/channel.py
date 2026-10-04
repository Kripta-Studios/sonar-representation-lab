"""Execute one frozen source-only Channel roster, retaining partial exposures."""

import argparse
import json
import os
import time
from pathlib import Path

from acquire import ROOT, download_ranges, extract, save_json
from budget import allocation_limits
from data import prepare_cache
from run_block import call
from runtime import LEDGER, LOCK, code_identity, file_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation", type=Path, default=Path("artifacts/spatial_motion_allocation.json"))
    parser.add_argument(
        "--resume", action="store_true", help="Resume only this identical frozen roster; retain all exposures"
    )
    args = parser.parse_args()
    freeze_path = Path("artifacts/freeze.json")
    frozen = json.loads(freeze_path.read_text())
    if frozen["status"] != "frozen" or code_identity() != frozen["code_identity"]:
        raise ValueError("Models/code are not the declared frozen programme")
    if file_sha("PROTOCOL.md") != frozen["protocol_sha256"] or LOCK.exists():
        raise ValueError("Protocol changed or another GPU job is active")
    allocation_limits(LEDGER, args.allocation, phase="final", freeze=freeze_path)
    journal = Path("artifacts/channel_exposure.json")
    if journal.exists():
        if not args.resume:
            raise RuntimeError(
                "Prior Channel exposure exists; preserve it and explicitly resume the same block"
            )
        exposed = json.loads(journal.read_text())
        if exposed["freeze_sha256"] != file_sha(freeze_path):
            raise ValueError("Cannot resume a different roster after exposure")
        exposed.setdefault("resumes_utc", []).append(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    else:
        if args.resume:
            raise ValueError("There is no recorded block to resume")
        exposed = {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "fixed block recorded before Channel download/extraction/image access",
            "freeze_sha256": file_sha(freeze_path),
            "prior_exposure": frozen["prior_exposure"],
            "models": [model["run"] for model in frozen["models"]],
            "attempts": [],
        }
    save_json(journal, exposed)
    os.environ["SONAR_RESEARCH_ALLOCATION"] = str(args.allocation.resolve())
    os.environ["SONAR_RESEARCH_PHASE"] = "final"
    os.environ["SONAR_FINAL_FREEZE"] = str(freeze_path.resolve())
    os.environ["SONAR_ORIGINAL_BLOCK_OPERATION"] = "0"
    try:
        archive = download_ranges("channel.tar", ROOT)
        if not exposed.get("archive_extracted"):
            extract(archive, ROOT)
            exposed["archive_extracted"] = True
            exposed["archive_sha256"] = file_sha(archive)
            save_json(journal, exposed)
        if any(model["config"]["detector_size"] == 448 for model in frozen["models"]):
            cache = ROOT / "cache/channel-gray448.json"
            if not cache.exists():
                prepare_cache(ROOT, "channel")
            exposed["cache_manifest_sha256"] = file_sha(cache)
            save_json(journal, exposed)
        for model in frozen["models"]:
            prior = [row for row in exposed["attempts"] if row["run"] == model["run"]]
            if prior and prior[-1].get("status") == "completed":
                if not args.resume:
                    raise ValueError("Unexpected pre-existing completed evaluation")
                if file_sha(Path(prior[-1]["out"]) / "metrics.json") != prior[-1]["metrics_sha256"]:
                    raise ValueError("Saved completed evaluation changed")
                continue
            suffix = "channel" if not prior else f"channel-replay-{len(prior)}"
            out = Path("artifacts") / model["run"] / suffix
            if out.exists() and any(out.iterdir()):
                raise ValueError("Unrecorded prior output exists; preserve and reconcile it")
            attempt = {
                "run": model["run"],
                "out": str(out),
                "status": "starting",
                "checkpoint_sha256": model["checkpoint_sha256"],
                "replay": bool(prior),
                "reason": "identical computation after incomplete attempt"
                if prior
                else "fixed initial evaluation",
            }
            exposed["attempts"].append(attempt)
            save_json(journal, exposed)
            call(
                "train_detector.py",
                [
                    "predict",
                    "--split",
                    "channel",
                    "--batch",
                    8,
                    "--checkpoint",
                    model["checkpoint"],
                    "--out",
                    out,
                    "--allocation",
                    args.allocation,
                ],
                out / "console.log",
            )
            result = json.loads((out / "metrics.json").read_text())
            attempt.update({"status": "completed", "metrics_sha256": file_sha(out / "metrics.json")})
            save_json(journal, exposed)
            print("HELD_OUT_RESULT", model["run"], result["AP50"], result["AP50:95"], flush=True)
        exposed.update(
            {
                "status": "completed fixed held-out block",
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    except BaseException as error:
        exposed.update({"status": "interrupted fixed block; exposure retained", "error": str(error)})
        raise
    finally:
        save_json(journal, exposed)


if __name__ == "__main__":
    main()
