"""Reproducible acquisition of the NASA SMAP/MSL (Telemanom) benchmark.

Provenance (investigated 2026-11-11):
* The original Telemanom README pointed to ``s3-us-west-2.amazonaws.com/telemanom/data.zip``;
  that bucket now returns HTTP 403. The current README points to a Kaggle copy that needs an
  API key.
* ``labeled_anomalies.csv`` is still served from the canonical GitHub repo (khundman/telemanom).
* The Hugging Face dataset ``lxr11111/telemanom`` mirrors the original ``.npy`` files unchanged
  (``data/data/{train,test}/<chan>.npy``) next to Parquet conversions, under the original
  Caltech BSD-3-clause license.

Strategy: download the *original .npy files* from the mirror at a **pinned commit**, verify each
file against the SHA-256 recorded by the mirror's Git-LFS metadata, and cross-check the mirror's
label file byte-for-byte against the canonical GitHub copy. Files that already exist with the right
hash are skipped, so re-running is cheap.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.request
from pathlib import Path

from src.utils.reproducibility import get_logger

log = get_logger("download")

HF_REPO = "lxr11111/telemanom"
HF_REVISION = "d93d47d794b7ca2ca860b9157dbfbe29ad8e08d6"  # pinned mirror commit
HF_TREE_URL = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}?recursive=1"
HF_FILE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{{path}}"
GITHUB_LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
LABELS_FILE = "labeled_anomalies.csv"
# SHA-256 of labeled_anomalies.csv as verified on 2026-11-11 (GitHub master == pinned mirror copy).
# GitHub `master` is a moving target, so the content is pinned by hash instead of by commit.
LABELS_SHA256 = "057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539"
MANIFEST_FILE = "MANIFEST.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, timeout: int = 60, attempts: int = 4) -> bytes:
    """GET ``url`` with retries; truncated or failed transfers are retried with backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": "telemetry-anomaly-research/1.0"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URLs)
                return resp.read()
        except (OSError, http.client.HTTPException) as exc:  # URLError and IncompleteRead included
            if attempt == attempts:
                raise
            log.warning("Fetch failed (%s), retry %d/%d: %s", type(exc).__name__, attempt, attempts - 1, url)
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def build_manifest() -> list[dict]:
    """List the original .npy files on the pinned mirror commit with their size and SHA-256."""
    tree = json.loads(_fetch(HF_TREE_URL))
    entries = []
    for item in tree:
        path = item.get("path", "")
        if item.get("type") != "file" or not path.endswith(".npy"):
            continue
        parts = path.split("/")
        if len(parts) != 4 or parts[:2] != ["data", "data"] or parts[2] not in ("train", "test"):
            continue  # skip anything that is not data/data/{train,test}/<chan>.npy
        lfs = item.get("lfs") or {}
        if "oid" not in lfs:
            raise RuntimeError(f"Mirror entry without LFS hash, cannot verify: {path}")
        entries.append({"remote_path": path, "split": parts[2], "file": parts[3],
                        "sha256": lfs["oid"], "size": lfs["size"]})
    if not entries:
        raise RuntimeError("Mirror listing returned no .npy files - layout may have changed.")
    return sorted(entries, key=lambda e: (e["split"], e["file"]))


def download_dataset(raw_dir: Path, force: bool = False) -> Path:
    """Download + verify the benchmark into ``raw_dir`` (``train/``, ``test/``, labels, manifest)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / MANIFEST_FILE
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        log.info("Using existing manifest (%d files).", len(manifest["files"]))
    else:
        log.info("Listing files on mirror %s@%s", HF_REPO, HF_REVISION[:8])
        manifest = {"source": f"https://huggingface.co/datasets/{HF_REPO}", "revision": HF_REVISION,
                    "labels_source": GITHUB_LABELS_URL, "files": build_manifest()}

    n_downloaded = 0
    for entry in manifest["files"]:
        target = raw_dir / entry["split"] / entry["file"]
        if target.exists() and target.stat().st_size == entry["size"] and sha256_of(target) == entry["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _fetch(HF_FILE_URL.format(path=entry["remote_path"]))
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {entry['remote_path']} - refusing to save it.")
        target.write_bytes(data)
        n_downloaded += 1
    log.info("Telemetry files: %d downloaded, %d already present and verified.",
             n_downloaded, len(manifest["files"]) - n_downloaded)

    # Labels: canonical GitHub copy, cross-checked against the mirror's copy.
    labels_path = raw_dir / LABELS_FILE
    if force or not labels_path.exists() or sha256_of(labels_path) != LABELS_SHA256:
        canonical = _fetch(GITHUB_LABELS_URL)
        mirror = _fetch(HF_FILE_URL.format(path=LABELS_FILE))
        manifest["labels_match_mirror"] = canonical == mirror
        digest = hashlib.sha256(canonical).hexdigest()
        if digest != LABELS_SHA256:
            # Upstream changed: fall back to the pinned mirror copy if it still matches, else stop.
            if hashlib.sha256(mirror).hexdigest() != LABELS_SHA256:
                raise RuntimeError("Neither GitHub nor mirror labels match the pinned SHA-256; "
                                   "inspect the change before updating LABELS_SHA256.")
            log.warning("GitHub labels changed upstream; using the pinned mirror copy.")
            canonical, digest = mirror, LABELS_SHA256
        labels_path.write_bytes(canonical)
        manifest["labels_sha256"] = digest

    manifest_path.write_text(json.dumps(manifest, indent=1))
    return raw_dir
