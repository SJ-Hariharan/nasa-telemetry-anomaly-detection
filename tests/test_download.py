"""The downloader's progress callback and TLS verification (offline)."""
import hashlib
import json
import os
import subprocess
import sys

import pytest



def test_download_reports_progress_for_every_file(tmp_path, monkeypatch):
    import src.data.download as dl

    files = []
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
        data = f"{split}-bytes".encode()
        (tmp_path / split / "X-1.npy").write_bytes(data)
        files.append({"split": split, "file": "X-1.npy", "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "remote_path": f"{split}/X-1.npy"})
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": files}))
    labels = b"chan_id,spacecraft,anomaly_sequences,class,num_values\n"
    (tmp_path / dl.LABELS_FILE).write_bytes(labels)
    monkeypatch.setattr(dl, "LABELS_SHA256", hashlib.sha256(labels).hexdigest())
    monkeypatch.setattr(dl, "_fetch", lambda *a, **k: pytest.fail("no network access expected"))
    seen = []
    dl.download_dataset(tmp_path, progress=lambda d, t, m: seen.append((d, t, m)))
    assert [(d, t) for d, t, _ in seen] == [(1, 2), (2, 2)] and all("verified" in m for *_, m in seen)


def test_downloads_verify_tls_with_certifi_bundle():
    """Regression: python.org macOS Python has no system CA store; downloads must still verify certificates."""
    import ssl

    import certifi

    from src.data.download import ssl_context

    ctx = ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert ctx.cert_store_stats()["x509_ca"] > 0          # a CA bundle was loaded
    assert certifi.where()


def _copy(tmp_path, chans=("A-1",)):
    import numpy as np

    src = tmp_path / "kaggle" / "smap-msl"
    for split in ("train", "test"):
        (src / "data" / split).mkdir(parents=True)
        for c in chans:
            np.save(src / "data" / split / f"{c}.npy", np.zeros((5, 2), np.float32))
    (src / "labeled_anomalies.csv").write_text("chan_id,spacecraft,anomaly_sequences,class,num_values\n")
    return src


def _checksums(tmp_path, src, monkeypatch, tamper=False):
    from src.data import download

    files = {f"{s}/{p.name}": [download.sha256_of(p), p.stat().st_size]
             for s in ("train", "test") for p in (src / "data" / s).glob("*.npy")}
    ref = {"source": "test", "labels_sha256": download.sha256_of(src / "labeled_anomalies.csv"), "files": files}
    if tamper:
        ref["files"]["train/A-1.npy"][0] = "0" * 64
    f = tmp_path / "checksums.json"
    f.write_text(json.dumps(ref))
    monkeypatch.setattr(download, "CHECKSUMS", f)


def test_import_local_copy_finds_and_verifies(tmp_path, monkeypatch):
    from src.data.download import find_dataset_dir, import_local_copy

    src = _copy(tmp_path)
    _checksums(tmp_path, src, monkeypatch)
    found = find_dataset_dir(tmp_path / "kaggle")
    assert found == src / "data"
    st = import_local_copy(found, tmp_path / "raw")
    assert st["verified"] and (tmp_path / "raw" / "train" / "A-1.npy").exists()
    assert (tmp_path / "raw" / "labeled_anomalies.csv").exists()


def test_import_local_copy_refuses_changed_files(tmp_path, monkeypatch):
    from src.data.download import import_local_copy

    src = _copy(tmp_path)
    _checksums(tmp_path, src, monkeypatch, tamper=True)
    with pytest.raises(RuntimeError, match="differs"):
        import_local_copy(src / "data", tmp_path / "raw")
    assert not (tmp_path / "raw").exists()
    st = import_local_copy(src / "data", tmp_path / "raw", allow_unverified=True)
    assert not st["verified"] and st["changed"] == ["train/A-1.npy"]


def test_committed_checksums_cover_the_benchmark():
    from src.data.download import CHECKSUMS

    ref = json.loads(CHECKSUMS.read_text())
    assert len(ref["files"]) == 164 and len(ref["labels_sha256"]) == 64
