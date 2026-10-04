"""Package research artifacts for GitHub, or restore verified release assets."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "artifacts/github-publication-public"
MANIFEST = PROJECT / "github-artifact-manifest.json"
PART_LIMIT = 1536 * 1024**2
GITHUB_LIMIT = 2 * 1024**3


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def package():
    if MANIFEST.exists() or OUTPUT.exists():
        raise ValueError("Preserve an existing export; do not rebuild or overwrite it")
    files = sorted(
        p
        for p in (PROJECT / "artifacts").rglob("*")
        if p.is_file()
        and not p.relative_to(PROJECT / "artifacts").parts[0].startswith("github-publication")
        and not p.name.startswith("github-")
        and (
            not p.is_relative_to(PROJECT / "artifacts/sources")
            or p.suffix == ".json"
            or "LICENSE" in p.name
            or p.name.endswith("_revision.txt")
        )
    )
    vendor = PROJECT / "vendor/dinov2"
    files += sorted(
        p
        for p in vendor.rglob("*")
        if p.is_file()
        and ".git" not in p.relative_to(vendor).parts
        and "__pycache__" not in p.relative_to(vendor).parts
    )
    source_bytes = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(PROJECT).free < source_bytes + 2 * 1024**3:
        raise RuntimeError("Insufficient space for a conservative uncompressed export forecast")
    OUTPUT.mkdir()
    groups, current, size = [], [], 0
    for path in files:
        length = path.stat().st_size
        if current and size + length > PART_LIMIT:
            groups.append(current)
            current, size = [], 0
        current.append(path)
        size += length
    if current:
        groups.append(current)
    record = {
        "format": "independent zip volumes; extract each into the repository root",
        "repository": "Kripta-Studios/sonar-representation-lab",
        "release": "research-2026-10-04",
        "visibility": "public",
        "includes": "research artifacts and pinned Apache-2.0 DINOv2 source without .git or bytecode",
        "excludes": "raw CFC imagery, Python environment, publication working files and cached third-party webpages/reference copies; factual provenance, revision IDs and licenses are retained",
        "file_count": len(files),
        "original_bytes": source_bytes,
        "assets": [],
    }
    for index, group in enumerate(groups, 1):
        target = OUTPUT / f"research-artifacts-{index:03d}.zip"
        entries = []
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for path in group:
                before = path.stat()
                name = path.relative_to(PROJECT).as_posix()
                sha = digest(path)
                archive.write(path, name)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed while packaging: {name}")
                entries.append({"path": name, "bytes": before.st_size, "sha256": sha})
        if target.stat().st_size >= GITHUB_LIMIT:
            raise RuntimeError(f"Release asset exceeds the per-file limit: {target}")
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"Archive CRC verification failed: {target}")
        asset = {
            "name": target.name,
            "bytes": target.stat().st_size,
            "sha256": digest(target),
            "files": entries,
        }
        record["assets"].append(asset)
        print(
            json.dumps({"completed_volume": index, "volumes": len(groups), "bytes": asset["bytes"]}),
            flush=True,
        )
    record["compressed_bytes"] = sum(item["bytes"] for item in record["assets"])
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    shutil.copyfile(MANIFEST, OUTPUT / MANIFEST.name)
    print(json.dumps({"manifest": str(MANIFEST), "assets": len(groups), "bytes": record["compressed_bytes"]}))


def restore(assets, destination):
    destination = destination.resolve()
    record = json.loads(MANIFEST.read_text(encoding="utf8"))
    for asset in record["assets"]:
        path = assets / asset["name"]
        if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
            raise ValueError(f"Asset identity mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            expected = {row["path"]: row for row in asset["files"]}
            if set(archive.namelist()) != set(expected):
                raise ValueError(f"Archive membership differs: {path}")
            for name, row in expected.items():
                target = (destination / name).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"Archive path escapes destination: {name}")
                if target.exists():
                    if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                        raise ValueError(f"Preserve differing existing file: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".github-restore-partial")
                with archive.open(name) as source, temporary.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)
                if temporary.stat().st_size != row["bytes"] or digest(temporary) != row["sha256"]:
                    raise ValueError(f"Restored file identity mismatch: {name}")
                temporary.replace(target)
        print(f"Verified and restored {asset['name']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("package", "restore"))
    parser.add_argument("--assets", type=Path, default=OUTPUT)
    parser.add_argument("--destination", type=Path, default=PROJECT)
    args = parser.parse_args()
    package() if args.action == "package" else restore(args.assets, args.destination)
