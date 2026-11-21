import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from google.cloud import vision
from PIL import Image


def our_model(model_dir: Path):
    meta = json.loads((model_dir / "model.json").read_text())
    sess = ort.InferenceSession(str(model_dir / "model.onnx"))

    def predict(path: Path) -> str:
        img = Image.open(path).convert("L")
        w = max(16, round(img.width * meta["height"] / img.height))
        x = np.asarray(img.resize((w, meta["height"]), Image.BILINEAR), dtype=np.float32) / 255.0
        x = ((x - meta["mean"]) / meta["std"])[None, None]
        best, prev, out = sess.run(None, {"image": x})[0][0].argmax(-1), 0, []
        for i in best:
            if i != prev and i != 0:
                out.append(meta["charset"][i])
            prev = i
        return "".join(out)

    return predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/plates/val")
    p.add_argument("--model", default="artifacts")
    p.add_argument("-n", type=int, default=50)
    a = p.parse_args()

    files = sorted(Path(a.data).glob("*.jpg"))[: a.n]
    if not files:
        sys.exit(f"no images in {a.data}")
    client, ours = vision.ImageAnnotatorClient(), our_model(Path(a.model))
    hits_vision = hits_ours = 0
    for f in files:
        truth = f.name.split("_")[0]
        resp = client.text_detection(image=vision.Image(content=f.read_bytes()))
        if resp.error.message:
            sys.exit(f"Vision API error: {resp.error.message}")
        v = re.sub(r"[^A-Z0-9]", "", resp.full_text_annotation.text.upper()).removeprefix("IND")
        o = ours(f)
        hits_vision += v == truth
        hits_ours += o == truth
        print(f"{truth:10s} vision={v!r:14s} ours={o!r}")
    print(f"\nexact match on {len(files)} images:  Cloud Vision {hits_vision / len(files):.0%}"
          f"   |   our model {hits_ours / len(files):.0%}")


if __name__ == "__main__":
    main()
