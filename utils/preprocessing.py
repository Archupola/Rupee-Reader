"""
Image preprocessing utilities for Rupee Reader.

Keeps the pipeline dead simple and dependency-light: resize + normalize,
matching what MobileNetV2 expects.
"""

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)


def load_image(file) -> Image.Image:
    """Load an image from a path, file-like object, or PIL Image and
    return it as an RGB PIL Image."""
    if isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)
    return img.convert("RGB")


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize to a model-ready batch of shape (1, 224, 224, 3).

    Pixel values are kept in raw 0-255 range (not divided down) because the
    model itself applies `mobilenet_v2.preprocess_input` internally as its
    first layer -- matching the raw-pixel input that
    `image_dataset_from_directory` produces during training in train.py.
    """
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr
