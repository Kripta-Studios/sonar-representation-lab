"""Official CFC acquisition; no signed URL is persisted or reused."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("D:/sonar-representation-lab-data")
RECORD = "https://data.caltech.edu/records/g945x-41103"
FILES = {
    "coco_annotations_v1.1.zip": ("dc6f119e1618dc823672127ccc7a9448", 14_497_860, RECORD),
    "file_lists_v1.1.zip": ("450ec249a48972b837236f7322677626", 6_489_562, RECORD),
    "fish_counting_metadata.tar.gz": (
        "152286bd6f25f965aadf41e8a0c44140",
        54_569,
        "https://data.caltech.edu/records/1y23m-j8r69",
    ),
    "kenai.tar": ("2f6a20c066495af6315147dd66aa2475", 44_074_741_065, RECORD),
    "channel.tar": ("16db94bef9af773b725301811f695e13", 1_680_528_910, RECORD),
}


def request(url, headers=None):
    return urllib.request.urlopen(
        urllib.request.Request(
            url,
            headers={
                "User-Agent": "sonar-representation-lab/0.1",
                "Cache-Control": "no-cache",
                **(headers or {}),
            },
        ),
        timeout=90,
    )


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=lambda value: value.tolist()), encoding="utf8")
    tmp.replace(path)


def download(name, root=ROOT):
    expected_md5, expected_size, record = FILES[name]
    archive = root / "archives" / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    provenance = root / "provenance.json"
    records = json.loads(provenance.read_text()) if provenance.exists() else {}
    if archive.exists():
        sha, md5 = hashes(archive)
        if md5 != expected_md5:
            raise ValueError(f"Existing archive fails official MD5: {archive}")
        if records.get(name, {}).get("sha256") == sha:
            print(f"Already verified: {name}", flush=True)
            return archive
    partial = archive.with_suffix(archive.suffix + ".part")
    # Every retry resolves the original publisher URL anew.
    url = f"{record}/files/{urllib.parse.quote(name)}?download=1"
    for attempt in range(3):
        start = partial.stat().st_size if partial.exists() else 0
        try:
            with request(url, {"Range": f"bytes={start}-"} if start else {}) as r:
                ranged = r.status == 206 and start > 0
                if ranged and not r.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Incorrect Content-Range")
                if not ranged:
                    start = 0
                size = start + int(r.headers["Content-Length"])
                if expected_size and size != expected_size:
                    raise ValueError(f"Publisher size changed: {size} vs {expected_size}")
                # Full image tar + extraction + 20 GiB model/cache allowance, without deleting anything.
                required = (size - start) + (size + 20 * 1024**3 if name.endswith(".tar") else 512 * 1024**2)
                free = shutil.disk_usage(root).free
                if free < required:
                    raise OSError(f"Need {required / 1024**3:.1f} GiB, have {free / 1024**3:.1f} GiB")
                print(
                    f"{name}: HTTP {r.status}, {size} bytes, redirected host {urllib.parse.urlsplit(r.url).netloc}",
                    flush=True,
                )
                last = time.monotonic()
                written = start
                with partial.open("ab" if ranged else "wb") as f:
                    while chunk := r.read(8 * 1024**2):
                        f.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last > 20:
                            print(f"{name}: {written / size:.1%} ({written / 1024**3:.2f} GiB)", flush=True)
                            last = time.monotonic()
            if partial.stat().st_size != size:
                raise OSError("Truncated response")
            sha, md5 = hashes(partial)
            if md5 != expected_md5:
                raise ValueError(f"Official MD5 mismatch for {name}: {md5}")
            partial.replace(archive)
            records[name] = {
                "dataset_version": "CFC v1.1"
                if name != "fish_counting_metadata.tar.gz"
                else "official shared clip metadata from v1.0 record",
                "original_filename": name,
                "publisher_url": url,
                "size_bytes": size,
                "publisher_md5": expected_md5,
                "local_md5": md5,
                "sha256": sha,
                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_json(provenance, records)
            return archive
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"Failed {url}: {type(e).__name__}: {e}", flush=True)
            if attempt == 2:
                raise


def download_ranges(name, root=ROOT, workers=4):
    """Bounded official HTTP ranges for an interrupted large archive."""
    expected_md5, size, record = FILES[name]
    if size is None:
        raise ValueError("Exact publisher file size is required")
    archive = root / "archives" / name
    if archive.exists():
        return download(name, root)
    partial = archive.with_suffix(archive.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    required = size - start + size * 2 + 20 * 1024**3
    if shutil.disk_usage(root).free < required:
        raise OSError("Insufficient archive/extraction/cache space")
    url = f"{record}/files/{urllib.parse.quote(name)}?download=1"
    chunk_size = 128 * 1024**2
    ranges = [(s, min(size - 1, s + chunk_size - 1)) for s in range(start, size, chunk_size)]

    def fetch(bounds):
        lo, hi = bounds
        path = archive.parent / f"{name}.range-{lo}-{hi}"
        if path.exists() and path.stat().st_size == hi - lo + 1:
            return path
        for attempt in range(3):
            try:
                received = path.stat().st_size if path.exists() else 0
                begin = lo + received
                # A fresh original publisher request prevents an intervening cache from reusing a signed redirect.
                fresh_url = url + "&request_time=" + str(time.time_ns())
                with request(fresh_url, {"Range": f"bytes={begin}-{hi}"}) as r:
                    expected = f"bytes {begin}-{hi}/{size}"
                    if r.status != 206 or r.headers.get("Content-Range") != expected:
                        raise ValueError(f"Publisher did not honor exact range {expected}")
                    with path.open("ab") as f:
                        while data := r.read(1024**2):
                            f.write(data)
                if path.stat().st_size != hi - lo + 1:
                    raise OSError("Truncated bounded range")
                return path
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"Failed {url} range {lo}-{hi}: {type(e).__name__}: {e}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Range acquisition failed")

    # Keep only one window of workers in flight, so staging uses <=512 MiB.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool, partial.open("ab") as dest:
        for first in range(0, len(ranges), workers):
            paths = list(pool.map(fetch, ranges[first : first + workers]))
            for path in paths:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, dest, 8 * 1024**2)
                dest.flush()
                # Only remove this downloader's exact appended chunk, inside its archive directory.
                assert path.resolve().parent == archive.parent.resolve()
                path.unlink()
            print(
                f"{name}: {dest.tell() / size:.1%} ({dest.tell() / 1024**3:.2f} GiB), verified HTTP ranges",
                flush=True,
            )
    sha, md5 = hashes(partial)
    if md5 != expected_md5:
        raise ValueError(f"Official MD5 mismatch for {name}: {md5}")
    partial.replace(archive)
    provenance = root / "provenance.json"
    records = json.loads(provenance.read_text())
    records[name] = {
        "dataset_version": "CFC v1.1",
        "original_filename": name,
        "publisher_url": url,
        "size_bytes": size,
        "publisher_md5": expected_md5,
        "local_md5": md5,
        "sha256": sha,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transfer": "bounded official HTTP ranges; fresh publisher redirects",
    }
    save_json(provenance, records)
    return archive


def extract(archive, root=ROOT):
    dest = root / ("images" if archive.name.endswith(".tar") else "metadata")
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = (dest / info.filename).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError("Unsafe ZIP path")
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            required = sum(m.size for m in t)
            if shutil.disk_usage(dest).free < required + 10 * 1024**3:
                raise OSError("Insufficient extraction/cache space")
            t.extractall(dest, filter="data")
    print(f"Extracted {archive.name} into {dest}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["metadata", "kenai", "channel"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--extract", action="store_true")
    p.add_argument("--ranges", action="store_true", help="Bounded official ranges for large archive transfer")
    args = p.parse_args()
    names = list(FILES)[:3] if args.stage == "metadata" else [args.stage + ".tar"]
    for name in names:
        archive = download_ranges(name, args.root) if args.ranges else download(name, args.root)
        if args.extract:
            extract(archive, args.root)


if __name__ == "__main__":
    main()
