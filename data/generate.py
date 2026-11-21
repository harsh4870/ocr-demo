import argparse
import io
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

OUT_SIZE = (200, 50)
STATES = ("AP AR AS BR CG CH DL GA GJ HP HR JH JK KA KL LA MH ML MN MP MZ NL OD PB PY "
          "RJ SK TN TR TS UK UP WB").split()
SERIES = "ABCDEFGHJKLMNPQRSTUVWXYZ"
STYLES = [((245, 245, 240), (15, 15, 15), 0.6), ((250, 200, 20), (15, 15, 15), 0.25),
          ((20, 120, 60), (245, 245, 240), 0.15)]
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/verdanab.ttf",
]
FONTS = [f for f in FONT_CANDIDATES if Path(f).exists()]
TRAIN_FONTS, VAL_FONTS = (FONTS[:-1], FONTS[-1:]) if len(FONTS) > 1 else (FONTS, FONTS)


def random_plate(rng: random.Random) -> str:
    if rng.random() < 0.1:
        return f"{rng.randint(21, 25)}BH{rng.randint(0, 9999):04d}{''.join(rng.choices(SERIES, k=2))}"
    series = "".join(rng.choices(SERIES, k=rng.choice((1, 2, 2, 2))))
    return f"{rng.choice(STATES)}{rng.randint(1, 99):02d}{series}{rng.randint(1, 9999):04d}"


def spaced(label: str) -> str:
    return " ".join(re.findall(r"[A-Z]+|\d+", label))


def render(label: str, rng: random.Random, fonts: list[str]) -> Image.Image:
    bg, fg = rng.choices([(b, t) for b, t, _ in STYLES], weights=[p for *_, p in STYLES])[0]
    size = rng.randint(56, 72)
    font = ImageFont.truetype(rng.choice(fonts), size) if fonts else ImageFont.load_default(size)
    text = spaced(label)
    l, t, r, b = font.getbbox(text)
    pad_x, pad_y, strip = rng.randint(14, 30), rng.randint(10, 22), int(size * 0.55)
    w, h = (r - l) + 2 * pad_x + strip, (b - t) + 2 * pad_y

    plate = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(plate)
    d.rounded_rectangle((2, 2, w - 3, h - 3), radius=10, outline=fg, width=rng.randint(3, 6))
    d.rectangle((8, 8, 8 + strip - 6, h - 9), fill=(25, 60, 160))
    d.text((8 + (strip - 6) / 2, h / 2), "IND", fill=(240, 240, 240),
           font=font.font_variant(size=max(10, size // 4)), anchor="mm")
    d.text((strip + pad_x - l, pad_y - t), text, fill=fg, font=font)

    bumper = tuple(rng.randint(20, 200) for _ in range(3))
    scene = Image.new("RGB", (int(w * 1.3), int(h * 1.8)), bumper)
    ox, oy = (scene.width - w) // 2, (scene.height - h) // 2
    scene.paste(plate, (ox, oy))
    scene = scene.rotate(rng.uniform(-3.5, 3.5), resample=Image.BILINEAR, fillcolor=bumper)
    shear = rng.uniform(-0.15, 0.15)
    scene = scene.transform(scene.size, Image.AFFINE, (1, shear, -shear * scene.height / 2, 0, 1, 0),
                            resample=Image.BILINEAR, fillcolor=bumper)

    mx, my = int(w * 0.02), int(h * 0.12)
    box = (ox - mx - rng.randint(0, 2 * mx), oy - my - rng.randint(0, my),
           ox + w + mx + rng.randint(0, 2 * mx), oy + h + my + rng.randint(0, my))
    img = scene.crop(box)

    lo = rng.uniform(0.35, 1.0)
    img = img.resize((max(40, int(OUT_SIZE[0] * lo)), max(10, int(OUT_SIZE[1] * lo))), Image.BILINEAR)
    img = img.resize(OUT_SIZE, Image.BILINEAR).filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.1)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.55, 1.35))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.3))
    x = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.3:
        a = rng.randint(0, OUT_SIZE[0] - 40)
        x[:, a:a + rng.randint(30, 100)] *= rng.uniform(0.35, 0.7)
    x += np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(0, 12), x.shape)
    img = Image.fromarray(x.clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(35, 90))
    return Image.open(buf)


def generate(out: Path, n: int, rng: random.Random, fonts: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        label = random_plate(rng)
        render(label, rng, fonts).save(out / f"{label}_{i}.jpg", quality=95)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/plates")
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = random.Random(a.seed)
    print(f"train fonts: {[Path(f).stem for f in TRAIN_FONTS] or 'Pillow built-in'}, "
          f"held-out val font: {[Path(f).stem for f in VAL_FONTS] or 'Pillow built-in'}")
    generate(Path(a.out) / "train", a.train, rng, TRAIN_FONTS)
    generate(Path(a.out) / "val", a.val, rng, VAL_FONTS)
    print(f"wrote {a.train} train + {a.val} val images to {a.out}")
