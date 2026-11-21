import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from export import export
from model import CRNN, IMG_HEIGHT


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    w = max(16, round(img.width * IMG_HEIGHT / img.height))
    x = np.asarray(img.resize((w, IMG_HEIGHT), Image.BILINEAR), dtype=np.float32) / 255.0
    return (x - 0.5) / 0.5


def load_split(folder: Path):
    files = sorted(folder.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no .jpg files in {folder}")
    with ThreadPoolExecutor(32) as pool:
        images = list(pool.map(lambda f: preprocess(Image.open(f)), files))
    labels = [f.name.split("_")[0] for f in files]
    return torch.from_numpy(np.stack(images)).unsqueeze(1), labels


def greedy_decode(logits: torch.Tensor, charset: str) -> list[str]:
    out = []
    for seq in logits.argmax(-1).tolist():
        chars, prev = [], 0
        for i in seq:
            if i != prev and i != 0:
                chars.append(charset[i - 1])
            prev = i
        out.append("".join(chars))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/plates")
    p.add_argument("--out", default=os.environ.get("AIP_MODEL_DIR", "artifacts"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()
    out = Path(a.out.replace("gs://", "/gcs/"))
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.time()
    x_train, y_train = load_split(Path(a.data) / "train")
    x_val, y_val = load_split(Path(a.data) / "val")
    charset = "".join(sorted(set("".join(y_train))))
    idx = {c: i + 1 for i, c in enumerate(charset)}
    print(f"loaded {len(y_train)} train / {len(y_val)} val in {time.time() - t0:.0f}s, "
          f"image {tuple(x_train.shape[1:])}, charset={charset!r}, device={device}")

    model = CRNN(len(charset) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * ((len(y_train) + a.batch - 1) // a.batch))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    history, best = [], -1.0

    for epoch in range(1, a.epochs + 1):
        model.train()
        perm, losses, t0 = torch.randperm(len(y_train)), [], time.time()
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            labels = [y_train[i] for i in b]
            targets = torch.tensor([idx[c] for t in labels for c in t])
            target_lens = torch.tensor([len(t) for t in labels])
            logits = model(x_train[b].to(device))
            log_probs = logits.log_softmax(-1).transpose(0, 1).cpu()
            input_lens = torch.full((len(b),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, input_lens, target_lens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            preds = [p for s in range(0, len(y_val), 256)
                     for p in greedy_decode(model(x_val[s:s + 256].to(device)).cpu(), charset)]
        acc = sum(p == t for p, t in zip(preds, y_val)) / len(y_val)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_acc": acc})
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  val_acc {acc:.3f}  ({time.time() - t0:.0f}s)"
              f"  e.g. {y_val[0]} -> {preds[0]}")
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "charset": charset}, out / "model.pt")

    (out / "metrics.json").write_text(json.dumps({"best_val_acc": best, "history": history}, indent=2))
    ckpt = torch.load(out / "model.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    export(model, charset, out)
    print(f"done. best val exact-match accuracy {best:.3f}; artifacts in {out}")


if __name__ == "__main__":
    main()
