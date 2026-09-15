# dataset/

**No real currency images are bundled** — banknote photographs aren't freely
redistributable, and collecting/labeling a set takes time.

## Option A — synthetic demo data (zero setup)

```
python setup_demo.py
```

Generates `dataset/<denomination>/demo_*.png` synthetic cards and trains on
them. Good for demonstrating the pipeline. **Will not recognise real notes.**

## Option B — real images (what you'd need for a real classifier)

```
dataset/
├── 10/
│   ├── img001.jpg
│   └── ...
├── 20/
├── 50/
├── 100/
├── 200/
└── 500/
```

- Folder name = denomination label, matching `model/class_names.txt`.
- 30+ images per class minimum; more is better.
- Vary lighting, background, angle, distance, and folded vs flat.
- Include both note faces.
- JPG or PNG; `train.py` resizes to 224×224 automatically.

Remove the synthetic images first so they don't pollute training:

```
rm -rf dataset/*/demo_*.png
python train.py
```

Generated images are gitignored — don't commit them.
