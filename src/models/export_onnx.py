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


def export_onnx(
    lit_module: TwoTowerLightningModule,
    output_path: str | Path,
    data_module: RecSysDataModule,
    batch_size: int = 4,
) -> None:
    """Export lit_module.model to ONNX and verify outputs match PyTorch (tol=1e-3).

    Supports both local paths and GCS URIs (gs://bucket/path/model.onnx).
    """
    model = lit_module.model.cpu().eval()

    dummy_user_ids = torch.randint(1, max(data_module.n_users, 2), (batch_size,))
    dummy_behavior = torch.randn(batch_size, data_module.user_behavior_dim)
    dummy_movie_ids = torch.randint(1, max(data_module.n_movies, 2), (batch_size,))
    dummy_meta = torch.randn(batch_size, data_module.movie_meta_dim)
    dummy_inputs = (dummy_user_ids, dummy_behavior, dummy_movie_ids, dummy_meta)

    # Export to a local temp file (avoids type issues with BytesIO in torch 2.7+)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_fd)
    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            tmp_path,
            opset_version=17,
            input_names=["user_ids", "user_behavior", "movie_ids", "movie_meta"],
            output_names=["score"],
            dynamic_axes={
                "user_ids": {0: "batch"},
                "user_behavior": {0: "batch"},
                "movie_ids": {0: "batch"},
                "movie_meta": {0: "batch"},
                "score": {0: "batch"},
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
        "movie_ids": dummy_movie_ids.numpy(),
        "movie_meta": dummy_meta.numpy(),
    }
    ort_out = sess.run(["score"], ort_inputs)[0]

    with torch.no_grad():
        pt_out = model(*dummy_inputs).numpy()

    max_diff = float(np.abs(pt_out - ort_out).max())
    if max_diff >= 1e-3:
        raise RuntimeError(f"ONNX vs PyTorch max diff {max_diff:.2e} exceeds 1e-3")

    print(f"ONNX export verified (max diff={max_diff:.2e}): {output_path}")
