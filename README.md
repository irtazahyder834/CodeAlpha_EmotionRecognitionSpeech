# Emotion Recognition from Speech

A deep learning project that classifies human emotions (happy, sad, angry,
calm, neutral, fearful, disgust, surprised) from speech audio using a
**CNN + LSTM** model built with **PyTorch**, trained on MFCC
features from the RAVDESS dataset. Includes an interactive Streamlit app
for live predictions.

Built as part of the **CodeAlpha Machine Learning Internship**.

## Project Overview

This project extracts MFCC (Mel-Frequency Cepstral Coefficient) time
sequences from speech recordings using `librosa`, trains a CNN+LSTM deep
learning model to classify the speaker's emotion, and serves predictions
through a Streamlit web app where users can upload a WAV file, play it
back, and view the predicted emotion along with a full probability
breakdown.

## Model Architecture

Each 3-second clip is represented as a `(130, 40)` sequence of MFCC frames
(time steps x MFCC coefficients). The Conv1D layers learn local
spectro-temporal patterns; the LSTM layer models how those patterns evolve
over the utterance:

```
Input (130 time steps x 40 MFCCs)
 -> Conv1D(64, kernel=5, relu) -> BatchNorm -> MaxPool -> Dropout(0.3)
 -> Conv1D(128, kernel=5, relu) -> BatchNorm -> MaxPool -> Dropout(0.3)
 -> LSTM(128) -> Dropout(0.4)
 -> Dense(64, relu) -> Dropout(0.3)
 -> Dense(num_classes, softmax)
```

- Framework: PyTorch
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Regularization: BatchNorm + Dropout + early stopping on a validation split

## Dataset

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song):
  24 professional actors, 8 emotions (neutral, calm, happy, sad, angry,
  fearful, disgust, surprised).
- Expected folder layout: `dataset/Actor_01/*.wav`, `dataset/Actor_02/*.wav`, etc.
- Optional: **TESS** and **EMO-DB** are also supported — place audio files
  in emotion-named subfolders and `utils.load_dataset_manifest` will fall
  back to using the folder name as the label.
- **Automatic offline demo fallback**: if `dataset/` contains no `.wav`
  files, `train.py` automatically generates an in-memory synthetic demo
  dataset (see `utils.generate_synthetic_dataset`) so the full pipeline —
  feature extraction, training, evaluation, and plots — still runs
  end-to-end out of the box. Add real RAVDESS audio and re-run for
  authentic results; `screenshots/classification_report.txt` clearly
  states which mode was used.

## Feature Extraction

For each audio clip (resampled to 22.05kHz, padded/truncated to 3 seconds),
`librosa` extracts 40 MFCC coefficients per frame, producing a
`(time_steps, 40)` sequence that is padded/truncated to a fixed 130 frames
before being fed to the CNN+LSTM model. Waveform, MFCC, and Mel Spectrogram
plots are also generated for a sample clip for visualization.

## Project Structure

```
Emotion_Recognition/
│
├── dataset/                             # (optional — place RAVDESS/TESS/EMO-DB audio here)
├── models/                              # Saved trained model (.pt) + label encoder
├── screenshots/                         # Auto-generated plots & reports
├── app.py                               # Streamlit GUI
├── train.py                             # Training pipeline
├── predict.py                           # Inference utilities / CLI
├── utils.py                             # Feature extraction & visualization helpers
├── requirements.txt
├── README.md
└── emotion_recognition.ipynb
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/CodeAlpha_EmotionRecognition.git
cd CodeAlpha_EmotionRecognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional but recommended) Download the RAVDESS dataset and place
#    actor folders inside dataset/. If skipped, train.py automatically
#    uses a synthetic demo dataset instead.

# 4. Train the model
python train.py --dataset_dir dataset

# 5. Launch the Streamlit app
streamlit run app.py
```

## Results

After running `train.py`, the following are generated in `screenshots/`:

- `waveform.png`, `mfcc.png`, `spectrogram.png` — sample audio visualizations
- `emotion_distribution.png` — class balance across the dataset
- `training_history.png` — training/validation loss & accuracy curves
- `confusion_matrix.png` — confusion matrix across all emotions
- `classification_report.txt` — precision, recall, F1-score, accuracy

The included training run used the offline synthetic fallback dataset
(since no RAVDESS audio was present) and reached 100% test accuracy on
that synthetic data — expected, since the synthetic waveforms are
cleanly separable by design. **Swap in real RAVDESS/TESS/EMO-DB audio in
`dataset/` and re-run `train.py` for a realistic accuracy figure**
(typically 60-75% test accuracy on RAVDESS with this architecture).

## Screenshots

See the `screenshots/` folder for generated waveform, MFCC, spectrogram,
training curves, and confusion matrix visualizations from an actual
training run included in this package.

## Future Improvements

- Add support for TESS and EMO-DB datasets in a unified training run.
- Apply data augmentation (pitch shift, time stretch, noise injection).
- Try attention layers or a pretrained audio embedding (e.g. wav2vec 2.0)
  as a feature extractor for a further accuracy boost.


### Repository Name
`CodeAlpha_EmotionRecognitionFromSpeech`
