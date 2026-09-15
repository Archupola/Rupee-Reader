# Rupee Reader

**AI-Based Indian Currency Recognition & Voice Assistant**

An accessibility-focused prototype that classifies currency denomination images and announces the result aloud.

> ⚠️ **Read the [Honest Limitations](#honest-limitations) section before demoing or submitting this.** The bundled demo model is trained on *synthetic placeholder images*, not on photographs of real banknotes.

## Problem Statement

Visually impaired and low-vision users may face difficulty identifying Indian currency denominations visually.

## Solution

Rupee Reader uses AI-based image classification to recognize a currency denomination and delivers the result both visually (large text + confidence score) and audibly (speech synthesis), so the denomination can be identified without relying on sight.

## Features

- Denomination classification across 6 classes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)
- CNN image classification — MobileNetV2 transfer learning, with an offline fallback
- Confidence score plus a full per-class score breakdown
- Automatic voice output (no button hunting required for accessibility)
- Image preprocessing pipeline
- **Zero-setup demo** — needs no dataset and no physical currency notes
- Runs fully locally; no paid APIs, no cloud services

## Honest Limitations

This is the part that matters most:

- **No real-currency dataset is bundled.** Photographs of banknotes aren't freely redistributable, so none are included.
- Instead, `utils/synthetic.py` generates **synthetic demo cards** — coloured rectangles with a denomination numeral, a per-class stripe pattern, and the word "DEMO" stamped on them. They are *not* reproductions of banknote designs.
- The classifier is then **genuinely trained** on those synthetic images. Training is real, the accuracy figure it reports is real — but it describes **only the synthetic demo set**.
- **The demo model will not reliably recognize photographs of real banknotes.** It has never seen one. The app labels this in its UI on every prediction.
- To build something that works on real notes, collect real images and retrain — see [Dataset](#dataset).
- This project recognizes **denomination only**. It does **not** authenticate currency, detect counterfeits, or offer any accuracy guarantee.

## Technologies

- Python 3.11 (3.10–3.12 supported)
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- Pillow / OpenCV
- Streamlit
- Web Speech API (browser-native text-to-speech)

## How It Works

```
Image (demo sample / upload / camera)
        ↓
Preprocessing — resize to 224×224, RGB
        ↓
MobileNetV2 (frozen ImageNet base) + trained dense head
        ↓
Softmax over 6 denominations
        ↓
Denomination + confidence score
        ↓
Voice output (browser speech synthesis)
```

## Supported Denominations

₹10 · ₹20 · ₹50 · ₹100 · ₹200 · ₹500

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start (no dataset needed)

```bash
python setup_demo.py && streamlit run app.py
```

`setup_demo.py` generates the synthetic dataset, generates held-out sample images, and trains the classifier (roughly 1–3 minutes on a laptop CPU). Then the app launches ready to demo.

**Alternative:** just run `streamlit run app.py` straight away. If no trained model exists, the app shows a one-click **"Generate demo data & train model"** button that does the same thing from the UI.

## Usage

1. Launch the app (above).
2. Open the **Demo samples** tab and pick a denomination — these are held out from training.
3. Click **Recognize Currency**.
4. The denomination and confidence appear, and the result is spoken automatically.
5. Use **📊 All class scores** to see the full softmax distribution, or **🔊 Speak again** to repeat the audio.

Command-line inference also works:

```bash
python predict.py sample_images/sample_demo_500.png
```

## Dataset

To train on real images instead of synthetic ones, arrange them like this:

```
dataset/
├── 10/    ├── 50/    ├── 200/
├── 20/    ├── 100/   └── 500/
```

One folder per denomination, folder names matching `model/class_names.txt`. Aim for 30+ images per class with varied lighting, angles, and backgrounds. Details in `dataset/README.md`.

Then delete the synthetic images and retrain:

```bash
rm -rf dataset/*/demo_*.png
python train.py
```

`train.py` records whether its training data was synthetic or real in `model/model_info.json`, and the app adjusts its labeling accordingly.

## Training

```bash
python train.py              # default 6 epochs
python train.py --epochs 15  # train longer
```

The preferred path is MobileNetV2 transfer learning with ImageNet weights (downloaded on first run, ~9 MB). If those weights can't be downloaded — no internet, blocked network — `train.py` automatically falls back to a small CNN trained from scratch, which works fully offline. Either way you get a working trained model; the architecture used is recorded in `model/model_info.json`.

## Project Structure

```
rupee-reader/
├── app.py                 # Streamlit UI
├── setup_demo.py          # one-command demo setup (generate + train)
├── train.py               # training script
├── predict.py             # inference (importable + CLI)
├── requirements.txt
├── model/
│   ├── class_names.txt
│   └── README.md
├── dataset/README.md
├── utils/
│   ├── preprocessing.py   # resize / normalize
│   ├── speech.py          # denomination → spoken phrase
│   └── synthetic.py       # synthetic demo image generator
├── sample_images/README.md
└── screenshots/README.md
```

## Future Enhancements

- Train on a real, varied photographic dataset of Indian banknotes (the key next step)
- Real-time continuous camera recognition
- Multilingual voice output (Hindi and other Indian languages)
- TensorFlow Lite conversion for on-device mobile deployment
- Additional denominations (₹2000, older note series)
- Improved robustness to lighting, folding, partial occlusion, and cluttered backgrounds
- Haptic feedback and screen-reader-optimized markup

## Disclaimer

This project is an educational accessibility prototype and is **not** a currency authentication or counterfeit detection system. It does not verify authenticity, does not detect counterfeits, and makes no guarantee of recognition accuracy. Do not rely on it for financial decisions or for identifying real currency. Denomination recognition and currency authentication are entirely different problems; this project addresses only the former, and only as a prototype.

## License

MIT — see [LICENSE](LICENSE).
