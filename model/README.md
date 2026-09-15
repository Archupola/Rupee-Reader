# model/

- `class_names.txt` — one denomination label per line, in training order.
- `currency_model.h5` — the trained classifier. **Not bundled**; created by
  `python setup_demo.py` or `python train.py`. Gitignored, since trained
  weights are regenerable build artifacts.
- `model_info.json` — provenance written at training time: number of images,
  epochs, final validation accuracy, architecture used, and crucially
  `synthetic_training_data` (true when trained on generated demo cards).
  `app.py` reads this to label its predictions honestly.

If `currency_model.h5` is absent, the app shows a one-click setup screen
instead of guessing.

The reported validation accuracy describes whatever data the model was
trained on. When that data is synthetic, a high number means "it learned the
synthetic cards well" — not that it recognises real banknotes.
