"""Archive the exact declared run source without changing a running experiment."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf8"))
    expected = {**config["code_identity"], "PROTOCOL.md": config["protocol_sha256"]}
    raw = {name: Path(name).read_bytes() for name in expected}
    for name, digest in expected.items():
        if hashlib.sha256(raw[name]).hexdigest() != digest:
            raise ValueError(f"Live source no longer matches this run: {name}")
    target = args.run / "source_snapshot.zip"
    if target.exists():
        with zipfile.ZipFile(target) as existing:
            assert all(existing.read(name) == value for name, value in raw.items())
        print("Verified existing source snapshot:", target)
        return
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in raw.items():
            archive.writestr(name, value)
        archive.writestr("requirements-lock.txt", Path("requirements-lock.txt").read_bytes())
    print("Saved exact declared source:", target)


if __name__ == "__main__":
    main()
