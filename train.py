"""
train.py - Train the Rupee Reader currency classifier.

Uses MobileNetV2 (ImageNet weights, frozen base) plus a small trainable
classification head. If ImageNet weights are unavailable, it falls back
to a small offline CNN.

Usage:
    python train.py
    python setup_demo.py

Expects images under dataset/<denomination>/*.(jpg|png).

Writes:
    model/currency_model.keras
    model/class_names.txt
    model/model_info.json
"""

import argparse
import json
import os
import sys

DATASET_DIR = "dataset"
MODEL_DIR = "model"

# Use modern Keras format to avoid H5 serialization problems.
MODEL_PATH = os.path.join(MODEL_DIR, "currency_model.keras")

CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DEFAULT_EPOCHS = 6

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(dataset_dir: str):
    found = []

    if not os.path.isdir(dataset_dir):
        return found

    for root, _dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                found.append(os.path.join(root, f))

    return found


def looks_synthetic(paths) -> bool:
    names = [os.path.basename(p) for p in paths]
    return bool(names) and all(n.startswith("demo_") for n in names)


def _build_mobilenet_model(tf, num_classes: int):
    from tensorflow.keras import layers, models

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def _build_small_cnn(tf, num_classes: int):
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=IMG_SIZE + (3,)),

            layers.Rescaling(1.0 / 255),

            layers.Conv2D(
                16,
                3,
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(),

            layers.Conv2D(
                32,
                3,
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(),

            layers.Conv2D(
                64,
                3,
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(),

            layers.Conv2D(
                64,
                3,
                activation="relu",
                padding="same"
            ),

            layers.GlobalAveragePooling2D(),

            layers.Dropout(0.2),

            layers.Dense(
                128,
                activation="relu"
            ),

            layers.Dense(
                num_classes,
                activation="softmax"
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_model(num_classes: int):
    import tensorflow as tf

    try:
        return (
            _build_mobilenet_model(tf, num_classes),
            True,
        )

    except Exception as exc:

        print(
            "\n[warn] Could not download MobileNetV2 ImageNet "
            f"weights ({type(exc).__name__}). "
            "Using the offline CNN fallback.\n"
        )

        return (
            _build_small_cnn(tf, num_classes),
            False,
        )


def main(epochs: int = DEFAULT_EPOCHS):
    import tensorflow as tf

    images = list_images(DATASET_DIR)

    if len(images) < 12:
        print(
            f"Only found {len(images)} image(s) under "
            f"'{DATASET_DIR}/'.\n\n"
            "Run:\n"
            "    python setup_demo.py\n"
        )

        sys.exit(1)

    synthetic = looks_synthetic(images)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = list(train_ds.class_names)

    print(
        "Detected classes:",
        class_names
    )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )

    model, used_imagenet = build_model(
        num_classes=len(class_names)
    )

    effective_epochs = (
        epochs
        if used_imagenet
        else max(epochs, 20)
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=effective_epochs,
    )

    val_acc = float(
        history.history
        .get("val_accuracy", [float("nan")])[-1]
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # IMPORTANT:
    # Save using modern Keras format.
    model.save(MODEL_PATH)

    with open(
        CLASS_NAMES_PATH,
        "w"
    ) as f:

        f.write(
            "\n".join(class_names)
            + "\n"
        )

    with open(
        INFO_PATH,
        "w"
    ) as f:

        json.dump(
            {
                "trained": True,
                "synthetic_training_data": synthetic,
                "num_training_images": len(images),
                "epochs": effective_epochs,
                "final_val_accuracy": val_acc,
                "class_names": class_names,
                "imagenet_weights": used_imagenet,
                "architecture": (
                    "MobileNetV2 (frozen ImageNet base) + dense head"
                    if used_imagenet
                    else "Small CNN trained from scratch "
                         "(offline fallback)"
                ),
            },
            f,
            indent=2,
        )

    print(
        f"\nSaved model      -> {MODEL_PATH}"
    )

    print(
        f"Saved class list -> {CLASS_NAMES_PATH}"
    )

    print(
        f"Saved info       -> {INFO_PATH}"
    )

    print(
        "Final validation accuracy on this dataset: "
        f"{val_acc * 100:.1f}%"
    )

    if synthetic:

        print(
            "\nNOTE: this model was trained on SYNTHETIC "
            "demo images, not photographs of real banknotes."
        )

        print(
            "The accuracy above describes only the synthetic "
            "demo set and does not represent real-currency accuracy."
        )

    else:

        print(
            "\nNOTE: with a small real dataset, real-world "
            "accuracy will be limited."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the Rupee Reader classifier."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    args = parser.parse_args()

    main(
        epochs=args.epochs
    )