"""
predict.py - Currency denomination inference.

Loads the trained Keras model from model/currency_model.keras and reports the
predicted denomination plus a confidence score. Also reads
model/model_info.json so callers know whether the model was trained on
synthetic demo images or on real photographs.

CLI usage:
    python predict.py sample_images/sample_demo_500.png

If no model exists yet, run:
    python setup_demo.py
"""

import json
import os
import sys

import numpy as np

from utils.preprocessing import load_image, preprocess_image

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "currency_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")


def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def load_model_info() -> dict:
    if os.path.exists(INFO_PATH):
        try:
            with open(INFO_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def model_available() -> bool:
    return os.path.exists(MODEL_PATH)


class Predictor:
    """Loads the trained classifier and runs inference on images."""

    def __init__(self):
        if not model_available():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run 'python setup_demo.py' "
                "to generate synthetic demo data and train a demo model, or "
                "'python train.py' if you have your own images in dataset/."
            )

        import tensorflow as tf

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        self.class_names = load_class_names()
        self.info = load_model_info()

    @property
    def synthetic(self) -> bool:
        """True when the loaded model was trained on synthetic demo images."""
        return bool(self.info.get("synthetic_training_data", False))

    def predict(self, image):
        """image: path, file-like, or PIL Image.

        Returns (label, confidence, all_scores) where all_scores is a list of
        (label, probability) sorted high to low.
        """
        img = load_image(image)
        batch = preprocess_image(img)

        preds = self.model.predict(batch, verbose=0)[0]

        idx = int(np.argmax(preds))
        label = self.class_names[idx]
        confidence = float(preds[idx])

        all_scores = sorted(
            zip(
                self.class_names,
                (float(p) for p in preds)
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return label, confidence, all_scores


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.png")
        sys.exit(1)

    predictor = Predictor()
    label, confidence, _scores = predictor.predict(sys.argv[1])

    if predictor.synthetic:
        print(
            "[PROTOTYPE] Model trained on synthetic demo images only -- "
            "not validated on real banknotes."
        )

    print(f"Predicted denomination: Rs. {label}")
    print(f"Confidence: {confidence * 100:.1f}%")


if __name__ == "__main__":
    main()