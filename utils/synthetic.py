"""
utils/synthetic.py - Synthetic "demo note" image generator.

IMPORTANT / HONESTY NOTE
------------------------
These images are NOT pictures of real Indian banknotes and are not
reproductions of any banknote design. They are plainly synthetic
placeholder cards: a coloured rectangle, a denomination numeral, and a
per-denomination geometric stripe pattern, each stamped with the word
"DEMO".

They exist for one reason: so the full Rupee Reader pipeline
(image -> preprocessing -> CNN -> denomination -> confidence -> speech)
can be genuinely trained and demonstrated on a laptop with no dataset and
no physical cash. A model trained on these WILL classify these synthetic
cards, and will NOT work on photographs of real currency.
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont

# Distinct base colours + stripe counts give the CNN real, learnable visual
# structure per class. Colours are loosely inspired by common note palettes
# but the images are deliberately abstract, not note replicas.
DENOM_STYLES = {
    "10": {"bg": (196, 148, 92), "fg": (60, 35, 10), "stripes": 1},
    "20": {"bg": (214, 176, 66), "fg": (60, 45, 5), "stripes": 2},
    "50": {"bg": (96, 168, 176), "fg": (10, 45, 50), "stripes": 3},
    "100": {"bg": (128, 128, 190), "fg": (25, 25, 70), "stripes": 4},
    "200": {"bg": (226, 152, 70), "fg": (70, 35, 5), "stripes": 5},
    "500": {"bg": (138, 132, 120), "fg": (35, 32, 28), "stripes": 6},
}

NOTE_SIZE = (420, 200)  # roughly banknote-ish aspect ratio

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]


def _font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_demo_note(denomination: str, seed: int = 0) -> Image.Image:
    """Build one synthetic demo card for `denomination`.

    `seed` varies colour jitter, rotation, stripe offset and noise so a
    training set has useful variety instead of identical copies.
    """
    rng = random.Random(f"{denomination}-{seed}")
    style = DENOM_STYLES[str(denomination)]

    jitter = lambda c: max(0, min(255, c + rng.randint(-22, 22)))  # noqa: E731
    bg = tuple(jitter(c) for c in style["bg"])
    fg = style["fg"]

    w, h = NOTE_SIZE
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    # Per-denomination stripe pattern (a second learnable cue besides colour).
    stripe_gap = h // (style["stripes"] + 1)
    offset = rng.randint(-6, 6)
    for i in range(1, style["stripes"] + 1):
        y = i * stripe_gap + offset
        draw.line([(16, y), (w - 16, y)], fill=fg, width=3)

    draw.rectangle([6, 6, w - 7, h - 7], outline=fg, width=4)

    # Denomination numeral, large and centred.
    numeral = str(denomination)
    f_big = _font(rng.choice([84, 92, 100]))
    tb = draw.textbbox((0, 0), numeral, font=f_big)
    draw.text(
        ((w - (tb[2] - tb[0])) / 2, (h - (tb[3] - tb[1])) / 2 - 12),
        numeral,
        font=f_big,
        fill=fg,
    )

    # Small corner numeral + explicit DEMO marking.
    f_small = _font(26)
    draw.text((20, 16), numeral, font=f_small, fill=fg)
    draw.text((w - 110, h - 44), "DEMO", font=f_small, fill=fg)

    # Mild rotation so the classifier sees off-angle inputs too.
    angle = rng.uniform(-7, 7)
    img = img.rotate(angle, expand=True, fillcolor=(245, 245, 245))

    return img


def generate_dataset(dataset_dir: str = "dataset", per_class: int = 40) -> int:
    """Write `per_class` synthetic images per denomination into
    dataset/<denomination>/. Returns total images written."""
    total = 0
    for denom in DENOM_STYLES:
        out_dir = os.path.join(dataset_dir, denom)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(per_class):
            img = make_demo_note(denom, seed=i)
            img.save(os.path.join(out_dir, f"demo_{denom}_{i:03d}.png"))
            total += 1
    return total


def generate_samples(out_dir: str = "sample_images") -> int:
    """Write one held-out sample per denomination (different seeds from the
    training set) for quick manual testing in the app."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for denom in DENOM_STYLES:
        img = make_demo_note(denom, seed=9000 + count)
        img.save(os.path.join(out_dir, f"sample_demo_{denom}.png"))
        count += 1
    return count
