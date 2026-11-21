import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model import CRNN, IMG_HEIGHT


def export(model: CRNN, charset: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy = torch.randn(1, 1, IMG_HEIGHT, 128)
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "steps"}},
        opset_version=17, dynamo=False,
    )
    (out / "model.json").write_text(json.dumps({
        "charset": [""] + list(charset),
        "height": IMG_HEIGHT, "mean": 0.5, "std": 0.5,
    }, indent=2))

    x = torch.randn(2, 1, IMG_HEIGHT, 180)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(onnx_path)).run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    assert diff < 1e-3, f"ONNX mismatch: {diff}"
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), max diff vs torch {diff:.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/model.pt")
    p.add_argument("--out", default="artifacts")
    a = p.parse_args()
    ckpt = torch.load(a.ckpt, map_location="cpu")
    m = CRNN(len(ckpt["charset"]) + 1)
    m.load_state_dict(ckpt["state_dict"])
    export(m, ckpt["charset"], Path(a.out))
