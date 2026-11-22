"""Execute the Colab/Kaggle notebook locally in smoke mode (needs the downloaded benchmark; slow, ~3 min)."""
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (ROOT / "data" / "raw" / "telemanom" / "labeled_anomalies.csv").exists()


@pytest.mark.skipif(not HAS_DATA or not os.environ.get("RUN_NOTEBOOK"),
                    reason="set RUN_NOTEBOOK=1 with the benchmark downloaded")
def test_pipeline_notebook_smoke_mode_runs_locally():
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(ROOT / "notebooks" / "run_pipeline.ipynb", as_version=4)
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
        out = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
        assert "failures: none" in out and "saved" in out
    finally:
        shutil.rmtree(ROOT / "reports" / "runs" / "telemanom_nb_smoke", ignore_errors=True)
        shutil.rmtree(ROOT / "models" / "telemanom_nb_smoke", ignore_errors=True)
        (ROOT / "configs" / "telemanom_nb_smoke.yaml").unlink(missing_ok=True)
        shutil.rmtree(ROOT / "outputs", ignore_errors=True)


def test_notebook_does_not_mistake_colab_for_kaggle():
    """Regression (found on Colab): Colab images contain /kaggle, so a path check misdetected Kaggle."""
    import nbformat

    nb = nbformat.read(ROOT / "notebooks" / "run_pipeline.ipynb", as_version=4)
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert 'Path("/kaggle").exists()' not in src
    assert 'IN_KAGGLE = not IN_COLAB and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))' in src
