"""One predeclared held-out evaluation block; preserve every exposure."""

import json
import time
from pathlib import Path

from acquire import ROOT, download_ranges, extract, save_json
from run_block import call


def main():
    frozen = json.loads(Path("artifacts/freeze.json").read_text())
    journal = Path("artifacts/channel_exposure.json")
    if journal.exists():
        raise RuntimeError("Channel exposure already recorded; preserve it and label any repeat explicitly")
    archive = download_ranges("channel.tar", ROOT)
    extract(archive, ROOT)
    save_json(
        journal,
        {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "one fixed evaluation block started",
            "freeze": "artifacts/freeze.json",
            "models": [m["run"] for m in frozen["models"]],
        },
    )
    for model in frozen["models"]:
        out = Path("artifacts") / model["run"] / "channel"
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
            ],
            out / "console.log",
        )
        result = json.loads((out / "metrics.json").read_text())
        print("HELD_OUT_RESULT", model["run"], result["AP50"], result["AP50:95"], flush=True)
    exposed = json.loads(journal.read_text())
    exposed.update(
        {
            "status": "completed fixed held-out block",
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    save_json(journal, exposed)


if __name__ == "__main__":
    main()
