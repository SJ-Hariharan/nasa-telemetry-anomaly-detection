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
import ssl
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


def ssl_context() -> ssl.SSLContext:
    """TLS context that verifies certificates with certifi's CA bundle when available.

    The python.org macOS installers ship without system CA certificates until "Install Certificates.command" is run,
    so the default context fails with CERTIFICATE_VERIFY_FAILED on a fresh setup. certifi (installed with the
    requirements) gives the same verification without that manual step. Verification is never disabled."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _fetch(url: str, timeout: int = 60, attempts: int = 4) -> bytes:
    """GET ``url`` with retries; truncated or failed transfers are retried with backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": "telemetry-anomaly-research/1.0"})
    ctx = ssl_context()
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310 (fixed https URLs)
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


def download_dataset(raw_dir: Path, force: bool = False, progress=None) -> Path:
    """Download + verify the benchmark into ``raw_dir`` (``train/``, ``test/``, labels, manifest).

    ``progress``: optional callable ``(done, total, message)`` called after each telemetry file is checked or
    downloaded (used by the dashboard's progress bar). It never changes what is downloaded or verified."""
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
    total = len(manifest["files"])
    for i, entry in enumerate(manifest["files"], 1):
        target = raw_dir / entry["split"] / entry["file"]
        if target.exists() and target.stat().st_size == entry["size"] and sha256_of(target) == entry["sha256"]:
            if progress:
                progress(i, total, f"verified {entry['split']}/{entry['file']}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _fetch(HF_FILE_URL.format(path=entry["remote_path"]))
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {entry['remote_path']} - refusing to save it.")
        target.write_bytes(data)
        n_downloaded += 1
        if progress:
            progress(i, total, f"downloaded {entry['split']}/{entry['file']}")
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


CHECKSUMS = Path(__file__).with_name("checksums.json")   # committed copy of the verified manifest


def find_dataset_dir(search_root: Path) -> Path | None:
    """Locate a benchmark copy (a folder with labeled_anomalies.csv next to train/ and test/), e.g. /kaggle/input."""
    for labels in sorted(Path(search_root).rglob(LABELS_FILE)):
        for base in (labels.parent, labels.parent / "data"):
            if (base / "train").is_dir() and (base / "test").is_dir():
                return base
    return None


def import_local_copy(source: Path, raw_dir: Path, allow_unverified: bool = False) -> dict:
    """Copy an already-available benchmark (e.g. an attached Kaggle dataset) into ``raw_dir`` after checking every
    file against the committed checksums. Mismatches stop the import unless ``allow_unverified`` (then the manifest
    records them, and results from that data must not be called a reproduction)."""
    source = Path(source)
    labels = next((p for p in (source / LABELS_FILE, source.parent / LABELS_FILE) if p.exists()), None)
    if labels is None or not (source / "train").is_dir():
        raise FileNotFoundError(f"{source} does not look like the benchmark (need train/, test/, {LABELS_FILE})")
    ref = json.loads(CHECKSUMS.read_text())
    bad, missing = [], []
    for rel, (sha, size) in ref["files"].items():
        p = source / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size != size or sha256_of(p) != sha:
            bad.append(rel)
    labels_ok = sha256_of(labels) == ref["labels_sha256"]
    if (bad or missing or not labels_ok) and not allow_unverified:
        raise RuntimeError(f"Local copy differs from the verified benchmark: {len(bad)} changed, {len(missing)} missing"
                           f"{'' if labels_ok else ', labels changed'} (e.g. {(bad + missing)[:3]}). "
                           "Use the downloader instead, or allow_unverified=True for an explicitly unverified run.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for rel in ref["files"]:
        if (source / rel).exists():
            (raw_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (raw_dir / rel).write_bytes((source / rel).read_bytes())
    (raw_dir / LABELS_FILE).write_bytes(labels.read_bytes())
    status = {"source": "local copy", "verified": not (bad or missing or not labels_ok), "changed": bad,
              "missing": missing, "labels_sha256_ok": labels_ok, "reference": ref["source"],
              "files": [{"split": r.split("/")[0], "file": r.split("/")[1], "sha256": s, "size": z}
                        for r, (s, z) in ref["files"].items() if r not in missing]}
    (raw_dir / MANIFEST_FILE).write_text(json.dumps(status, indent=1))
    log.info("Imported %d files from a local copy (%s)", len(ref["files"]) - len(missing),
             "verified" if status["verified"] else "UNVERIFIED")
    return status
