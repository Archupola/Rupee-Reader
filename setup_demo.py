"""
setup_demo.py - One command to make the project demo-ready with no dataset
and no physical currency notes.

It:
  1. generates a synthetic "demo note" dataset (utils/synthetic.py),
  2. generates held-out sample images for testing,
  3. trains the MobileNetV2 transfer-learning classifier on it,
  4. saves model/currency_model.h5.

Usage:
    python setup_demo.py

Takes roughly 1-3 minutes on a normal laptop CPU.

HONESTY NOTE: the resulting model is trained on SYNTHETIC placeholder
images, not on photographs of real banknotes. It demonstrates the full
pipeline and will classify the synthetic demo cards; it will NOT reliably
recognise real currency. See README.md.
"""

import os

from utils.synthetic import generate_dataset, generate_samples

PER_CLASS = 40


def main():
    print("[1/3] Generating synthetic demo dataset...")
    n = generate_dataset("dataset", per_class=PER_CLASS)
    print(f"      wrote {n} images to dataset/")

    print("[2/3] Generating held-out sample images...")
    m = generate_samples("sample_images")
    print(f"      wrote {m} images to sample_images/")

    print("[3/3] Training MobileNetV2 classifier on the synthetic set...")
    import train

    train.main(epochs=6)

    model_path = os.path.join("model", "currency_model.h5")
    if os.path.exists(model_path):
        print("\nDone. Demo model ready at", model_path)
        print("Now run:  streamlit run app.py")
    else:
        print("\nTraining did not produce a model file; check the output above.")


if __name__ == "__main__":
    main()
