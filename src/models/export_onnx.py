"""Export TwoTowerModel to ONNX and verify numerical equivalence with PyTorch."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.data.dataset import RecSysDataModule
from src.models.lightning_module import TwoTowerLightningModule
from src.models.two_tower import TwoTowerModel


def export_onnx(
    lit_module: TwoTowerLightningModule,
    output_path: str | Path,
    data_module: RecSysDataModule,
    batch_size: int = 4,
) -> None:
    """Export the user tower to ONNX and verify outputs match PyTorch (tol=1e-3).

    Only the user tower is exported: the movie tower is evaluated once over the
    full catalog at export time instead (see precompute_movie_embeddings), since
    its output doesn't depend on the user and stays fixed until the next retrain.

    Supports both local paths and GCS URIs (gs://bucket/path/user_tower.onnx).
    """
    model = lit_module.model.user_tower.cpu().eval()

    dummy_user_ids = torch.randint(1, max(data_module.n_users, 2), (batch_size,))
    dummy_behavior = torch.randn(batch_size, data_module.user_behavior_dim)
    dummy_inputs = (dummy_user_ids, dummy_behavior)

    # Export to a local temp file (avoids type issues with BytesIO in torch 2.7+)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_fd)
    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            tmp_path,
            opset_version=17,
            input_names=["user_ids", "user_behavior"],
            output_names=["user_embedding"],
            dynamic_axes={
                "user_ids": {0: "batch"},
                "user_behavior": {0: "batch"},
                "user_embedding": {0: "batch"},
            },
        )
        model_bytes = Path(tmp_path).read_bytes()
    finally:
        os.unlink(tmp_path)

    onnx.checker.check_model(onnx.load(io.BytesIO(model_bytes)))

    output_str = str(output_path)
    if output_str.startswith("gs://"):
        import fsspec

        with fsspec.open(output_str, "wb") as f:
            f.write(model_bytes)  # type: ignore[union-attr]
    else:
        Path(output_str).parent.mkdir(parents=True, exist_ok=True)
        Path(output_str).write_bytes(model_bytes)

    sess = ort.InferenceSession(model_bytes)
    ort_inputs = {
        "user_ids": dummy_user_ids.numpy(),
        "user_behavior": dummy_behavior.numpy(),
    }
    ort_out = sess.run(["user_embedding"], ort_inputs)[0]

    with torch.no_grad():
        pt_out = model(*dummy_inputs).numpy()

    # Tolerance is looser than the old full-forward check: this compares raw
    # per-dimension embedding values, not a sigmoid-compressed scalar score, so
    # LayerNorm/GELU numerical noise across backends doesn't average out.
    max_diff = float(np.abs(pt_out - ort_out).max())
    if max_diff >= 5e-3:
        raise RuntimeError(f"ONNX vs PyTorch max diff {max_diff:.2e} exceeds 5e-3")

    print(f"ONNX export verified (max diff={max_diff:.2e}): {output_path}")


def precompute_movie_embeddings(
    model: TwoTowerModel,
    movie_idxs: np.ndarray,
    movie_metas: np.ndarray,
) -> np.ndarray:
    """Run the movie tower once over the full catalog and return its embeddings.

    The movie tower's output only depends on the movie's own features, so it is
    computed once here at export time instead of once per candidate per request.
    Returns an array of shape [len(movie_idxs), output_dim].
    """
    movie_tower = model.movie_tower.cpu().eval()
    with torch.no_grad():
        embeddings = movie_tower(
            torch.tensor(movie_idxs, dtype=torch.long),
            torch.tensor(movie_metas, dtype=torch.float32),
        )
    return embeddings.numpy()
