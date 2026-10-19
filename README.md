Rupee Reader

AI-Based Indian Currency Recognition & Voice Assistant

An accessibility-focused prototype that classifies currency denomination images and announces the result aloud.

⚠️ Important — Please Read: The current demo model is trained on synthetic placeholder images, not photographs of real Indian banknotes. The project demonstrates the complete AI/ML pipeline and application workflow, but the current model should not be considered a reliable real-currency recognition system. See the Honest Limitations section below.

Problem Statement

Visually impaired and low-vision users may face difficulty identifying Indian currency denominations visually.

Identifying different denominations can become difficult when users cannot easily distinguish the notes by sight. An AI-based image recognition system can assist by predicting the denomination from an image and providing the result through both visual and audio feedback.

Solution

Rupee Reader uses AI-based image classification to recognize a currency denomination and delivers the result both visually and audibly.

The application accepts an image through upload or camera input, preprocesses the image, passes it through a trained deep-learning model, predicts the denomination, displays the confidence score, and announces the result using text-to-speech.

The application runs locally and does not require paid APIs or cloud services.

Features

Denomination classification across 6 classes:

₹10

₹20

₹50

₹100

₹200

₹500

CNN-based image classification

MobileNetV2 transfer learning with an offline CNN fallback

Confidence score for predictions

Full per-class probability breakdown

Automatic voice output for accessibility

Image preprocessing pipeline

Image upload support

Camera input support

Built-in synthetic demo samples

Zero-setup demo — no dataset or physical currency notes required

Runs fully locally

No paid APIs

No cloud services required

Honest Limitations



This project is an educational accessibility prototype. The current version uses a synthetic demonstration dataset instead of photographs of real Indian banknotes.

Synthetic Demo Dataset

The repository does not include photographs of real banknotes.

Instead, utils/synthetic.py generates synthetic demonstration cards containing:

A denomination numeral

A class-specific stripe/pattern design

A DEMO label

Artificial backgrounds and layouts

These images are intentionally created as placeholders and are not reproductions of actual Indian banknote designs.

The Model Is Genuinely Trained

Although the dataset is synthetic, the machine-learning pipeline is real.

The project:

Generates the synthetic images.

Splits the images into training and validation data.

Trains an image-classification model.

Evaluates the model.

Saves the trained model.

Performs inference on held-out demo images.

Therefore, the reported validation accuracy is a genuine measurement of the model's performance on the generated synthetic dataset.

Synthetic Accuracy Does Not Equal Real-World Accuracy

A high validation accuracy on the synthetic dataset does not mean that the model can accurately recognize photographs of real Indian currency notes.

The current demo model has not been trained on photographs of real banknotes. As a result, it should not be expected to reliably recognize real ₹10, ₹20, ₹50, ₹100, ₹200, or ₹500 notes.

The application clearly communicates this limitation in its interface.

Real-World Dataset Required

To develop a system that works reliably on real currency, the synthetic dataset must be replaced with a sufficiently large and diverse real-world dataset.

The dataset should include variations in:

Lighting

Camera quality

Viewing angle

Note orientation

Backgrounds

Folded notes

Worn or damaged notes

Partial occlusion

Different distances from the camera

Different note conditions

The model would then need to be retrained and evaluated on an independent real-world test set.

Not a Counterfeit Detection System

Rupee Reader recognizes denomination only.

It does not:

Authenticate currency

Detect counterfeit notes

Verify security features

Determine whether a note is genuine

Provide any guarantee of accuracy

Denomination recognition and currency authentication are different problems. This project addresses denomination classification only.

Technologies

Python 3.11

TensorFlow / Keras

MobileNetV2

Transfer Learning

Pillow

OpenCV

NumPy

Streamlit

Web Speech API

Browser-native text-to-speech

How It Works

Image
(Demo Sample / Upload / Camera)
        ↓
Image Preprocessing
Resize to 224×224
Convert to RGB
        ↓
MobileNetV2
Frozen ImageNet Base
        ↓
Trained Dense Classification Head
        ↓
Softmax
        ↓
6 Denomination Classes
        ↓
Predicted Denomination
+ Confidence Score
        ↓
Visual Result + Voice Output

Supported Denominations

The current prototype supports:

₹10

₹20

₹50

₹100

₹200

₹500

Installation

Requirements

Recommended Python version:

Python 3.11

Python versions 3.10–3.12 are supported by the project dependencies.

Clone the Repository

git clone https://github.com/Archupola/Rupee-Reader.git
cd Rupee-Reader

Create a Virtual Environment

Windows

python -m venv .venv
.venv\Scripts\activate

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

Install Dependencies

pip install -r requirements.txt

Quick Start — No Dataset Needed

The project includes a synthetic demo-data generator, so you do not need physical currency notes or an external dataset to run the demonstration.

Run:

python setup_demo.py

The setup script:

Generates synthetic demonstration images.

Creates training and validation data.

Generates held-out sample images.

Trains the classifier.

Saves the trained model.

Saves class information and model metadata.

After training completes, launch the application:

streamlit run app.py

The application will open in your browser.

Alternative

You can also launch the application directly:

streamlit run app.py

If no trained model exists, the application provides a Generate Demo Data & Train Model option to create the synthetic dataset and train the demo model from the interface.

Usage

Launch the application:

streamlit run app.py

Open the Demo Samples section.

Select a denomination demo image.

Click Recognize Currency.

The application displays:

Predicted denomination

Confidence score

Full class probability distribution

The result is automatically spoken using browser text-to-speech.

Use Speak Again to repeat the voice output if required.

⚠️ For the current prototype, use the built-in Demo Samples when demonstrating the project. Real banknote photographs are not expected to be recognized reliably by the synthetic-data model.

Command-Line Inference

Command-line prediction is also supported.

Example:

python predict.py sample_images/sample_demo_500.png

The script performs preprocessing and returns the predicted denomination and confidence score.

Dataset

Current Demo Dataset

The current project automatically generates synthetic demonstration images.

The dataset follows this structure:

dataset/
├── 10/
├── 20/
├── 50/
├── 100/
├── 200/
└── 500/

The synthetic images are generated using:

utils/synthetic.py

The synthetic images are intentionally designed as demonstration placeholders rather than reproductions of actual Indian banknotes.

Training With Real Currency Images

For a real-world currency recognition system, replace the synthetic dataset with photographs of real currency notes.

Arrange the images as follows:

dataset/
├── 10/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 20/
│   ├── image1.jpg
│   └── ...
├── 50/
├── 100/
├── 200/
└── 500/

Each folder should contain images belonging to the corresponding denomination.

A useful dataset should contain variations in:

Lighting

Angles

Backgrounds

Note orientations

Camera distances

Note conditions

Aim for at least 30+ images per class for an initial experiment, with substantially more data recommended for meaningful real-world performance.

Additional dataset details are available in:

dataset/README.md

Retraining

After replacing the synthetic images with real images, retrain the model:

python train.py

The training process records dataset and model information in:

model/model_info.json

The application uses this metadata to distinguish between synthetic-demo and real-data training configurations.

Training

Default training:

python train.py

Train for more epochs:

python train.py --epochs 15

The preferred architecture is:

MobileNetV2
      ↓
Transfer Learning
      ↓
Classification Head
      ↓
Softmax
      ↓
6 Denomination Classes

MobileNetV2 uses pretrained ImageNet features as the base feature extractor.

If the pretrained ImageNet weights cannot be downloaded because of network restrictions, the training script automatically falls back to a lightweight CNN trained from scratch.

The architecture and training information are recorded in:

model/model_info.json

Model Files

After training, the generated model files are stored under:

model/

Important files include:

model/
├── currency_model.keras
├── class_names.txt
├── model_info.json
└── README.md

The trained model and generated synthetic dataset are excluded from version control where appropriate.

They can be recreated at any time using:

python setup_demo.py

Project Structure

Rupee-Reader/
│
├── app.py
│   └── Streamlit application
│
├── setup_demo.py
│   └── Generates synthetic demo data and trains the model
│
├── train.py
│   └── Model training pipeline
│
├── predict.py
│   └── Model inference and command-line prediction
│
├── requirements.txt
│   └── Python dependencies
│
├── model/
│   ├── class_names.txt
│   └── README.md
│
├── dataset/
│   └── README.md
│
├── sample_images/
│   └── README.md
│
├── screenshots/
│   └── README.md
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── speech.py
│   └── synthetic.py
│
├── .gitignore
├── LICENSE
└── README.md

Accessibility Focus

Rupee Reader is designed with accessibility as a core consideration.

The application provides:

Large prediction text

Clear visual feedback

Confidence information

Automatic voice output

Simple interaction flow

Browser-native text-to-speech

Local processing without a cloud service

The goal is to explore how computer vision and speech technologies can assist users who have difficulty visually identifying currency denominations.

Future Enhancements

Potential future improvements include:

Train on a large and diverse real-world Indian currency dataset

Real-time continuous camera recognition

Improved recognition under different lighting conditions

Recognition of rotated and partially occluded notes

Multilingual voice output

Hindi and other Indian language support

TensorFlow Lite conversion

On-device mobile deployment

Haptic feedback

Improved screen-reader compatibility

Additional denominations, including ₹2000 where appropriate

Improved robustness against cluttered backgrounds

Data augmentation and stronger regularization

Independent real-world evaluation

Precision, recall, F1-score and confusion-matrix evaluation

Mobile accessibility improvements

Current Prototype Status

Component

Status

Synthetic dataset generation

✅ Complete

Image preprocessing

✅ Complete

CNN classification

✅ Complete

MobileNetV2 transfer learning

✅ Complete

Offline CNN fallback

✅ Complete

Streamlit application

✅ Complete

Confidence display

✅ Complete

Voice output

✅ Complete

Demo samples

✅ Complete

Real currency dataset

⚠️ Not included

Real-world currency recognition

⚠️ Future work

Counterfeit detection

❌ Not implemented

Mobile deployment

🔜 Future enhancement

Disclaimer

Rupee Reader is an educational accessibility prototype.

The current demonstration model is trained using synthetic images and is not a production-ready real-currency recognition system.

This project:

Does not authenticate currency.

Does not detect counterfeit notes.

Does not verify security features.

Does not guarantee recognition accuracy.

Should not be relied upon for financial decisions.

Should not be used as the sole method of identifying real currency.

For real-world deployment, the model must be trained and evaluated using an appropriate real-currency dataset and tested extensively under realistic conditions.

License

This project is licensed under the MIT License.

See the LICENSE file for details.

Author

Archana Pola

GitHub: Archupola

Project Repository: Rupee Reader

Project Summary

Rupee Reader demonstrates an end-to-end AI/ML application workflow:

Data Generation / Collection
          ↓
Image Preprocessing
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Image Inference
          ↓
Denomination Prediction
          ↓
Accessibility Interface
          ↓
Voice Output

The current synthetic dataset makes the project reproducible without requiring physical currency or distributing real banknote photographs, while the documented limitations clearly distinguish the working prototype from a future real-world currency recognition system.
