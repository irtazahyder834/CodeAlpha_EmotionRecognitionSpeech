# 🎙️ Emotion Recognition from Speech

A deep learning project that classifies human emotions (**Happy, Sad, Angry, Calm, Neutral, Fearful, Disgust, Surprised**) from speech audio using a **CNN + LSTM** model built with **PyTorch**, trained on **MFCC** features extracted from the **RAVDESS** dataset.

The project also includes an interactive **Streamlit** web application for real-time emotion prediction from uploaded speech audio.

> 🚀 Built as part of the **CodeAlpha Machine Learning Internship**.

---

# 📌 Project Overview

This project extracts **MFCC (Mel-Frequency Cepstral Coefficients)** from speech recordings using **Librosa**, trains a **CNN + LSTM** deep learning model to classify emotions, and provides predictions through a **Streamlit** web application.

Users can:

- 🎤 Upload a WAV audio file
- ▶️ Listen to the uploaded audio
- 😊 Predict the speaker's emotion
- 📊 View confidence scores for all emotion classes

---

# 🧠 Model Architecture

Each **3-second** audio clip is converted into a **(130 × 40)** MFCC feature sequence.

```
Input (130 × 40 MFCC)

        │
        ▼
Conv1D (64, Kernel=5, ReLU)
        │
BatchNorm
        │
MaxPooling
        │
Dropout (0.3)
        │
        ▼
Conv1D (128, Kernel=5, ReLU)
        │
BatchNorm
        │
MaxPooling
        │
Dropout (0.3)
        │
        ▼
LSTM (128)
        │
Dropout (0.4)
        │
Dense (64, ReLU)
        │
Dropout (0.3)
        │
Dense (8 Classes, Softmax)
```

### ⚙️ Training Configuration

- 🧠 Framework: PyTorch
- ⚡ Optimizer: Adam
- 📉 Loss Function: Sparse Categorical Cross-Entropy
- 🛡️ Regularization:
  - Batch Normalization
  - Dropout
  - Early Stopping

---

# 🎵 Dataset

### 📂 RAVDESS

Ryerson Audio-Visual Database of Emotional Speech and Song

- 👥 24 Professional Actors
- 🎭 8 Emotion Classes
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised

Expected folder structure:

```
dataset/
│
├── Actor_01/
├── Actor_02/
├── Actor_03/
...
```

### ✅ Also Supported

- 🎤 TESS
- 🎤 EMO-DB

Simply place audio files inside emotion-named folders.

### 💡 Offline Demo Mode

If no dataset is detected, the project automatically creates a synthetic dataset so the complete training pipeline still runs successfully.

---

# 🎼 Feature Extraction

Each audio file is:

- 🎧 Resampled to **22.05 kHz**
- ✂️ Padded/Trimmed to **3 seconds**
- 📈 Converted into **40 MFCC coefficients**
- 📊 Fixed to **130 time frames**

The project also generates:

- 📈 Waveform
- 🎼 MFCC
- 🌈 Mel Spectrogram

---

# 📁 Project Structure

```text
CodeAlpha_EmotionRecognitionFromSpeech/
│
├──  dataset/
├──  models/
├──  screenshots/
├──  app.py
├──  train.py
├──  predict.py
├──  utils.py
├──  requirements.txt
├──  README.md
└──  emotion_recognition.ipynb
```

---

# 🚀 Installation

```bash
# Clone Repository
git clone https://github.com/<your-username>/CodeAlpha_EmotionRecognitionFromSpeech.git

cd CodeAlpha_EmotionRecognitionFromSpeech

# Install dependencies
pip install -r requirements.txt

# Train Model
python train.py --dataset_dir dataset

# Launch Streamlit App
streamlit run app.py
```

---

# 📊 Results

Running `train.py` automatically generates:

- 📈 Waveform
- 🎼 MFCC
- 🌈 Spectrogram
- 📊 Emotion Distribution
- 📉 Training History
- 🔥 Confusion Matrix
- 📄 Classification Report

The included demo uses a synthetic dataset and achieves **100% accuracy** because the generated samples are intentionally separable.

For realistic performance, replace the dataset with **RAVDESS**, **TESS**, or **EMO-DB**.

Expected accuracy:

**🎯 60–75% on the RAVDESS dataset**

---

# 🖼️ Screenshots

The **screenshots/** folder includes:

- 📈 Waveform
- 🎼 MFCC
- 🌈 Spectrogram
- 📉 Training Curves
- 🔥 Confusion Matrix
- 📄 Classification Report

---

# 🚀 Future Improvements

- ➕ Combine RAVDESS, TESS and EMO-DB datasets
- 🎧 Audio Data Augmentation
- 🧠 Attention Mechanism
- 🤖 Wav2Vec 2.0 Feature Extractor
- ☁️ Cloud Deployment
- 📱 Mobile-Friendly Interface

---

## 👨‍💻 Author

**Irtaza Hyder**

Machine Learning Intern at **CodeAlpha**

Bachelor of Science in Computer Science (BSCS)

---

⭐ If you found this project useful, don't forget to **Star** the repository!
