import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import speech_recognition as sr
import matplotlib.pyplot as plt

# Function to extract features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)  # Load audio
    # MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    # Combine features
    features = np.concatenate([mfccs_mean, chroma_mean, contrast_mean])
    return features

# Sample dataset (replace with real data for better results)
# Emotions: 0=Neutral, 1=Happy, 2=Sad, 3=Angry
emotions = ['neutral', 'happy', 'sad', 'angry']
# Dummy features (in practice, extract from multiple audio files)
X = np.random.rand(100, 26)  # 26 features
y = np.random.choice([0, 1, 2, 3], 100)  # Random labels for demo

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=emotions))

# Function to predict from audio file
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = np.max(prob)
    return emotions[prediction], confidence

# Function for live speech recognition (transcribes and analyzes)
def recognize_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak for emotion analysis...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Transcribed: {text}")
            # Basic text-based emotion (fallback; integrate with audio features for full SER)
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                emotion = "Happy"
            elif polarity < -0.1:
                emotion = "Sad"
            else:
                emotion = "Neutral"
            print(f"Estimated Emotion: {emotion} (Polarity: {polarity:.2f})")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Speech service error")

# Example usage
# For file: Replace 'sample.wav' with your audio file path
# emotion, conf = predict_emotion('sample.wav')
# print(f"Predicted Emotion: {emotion} (Confidence: {conf:.2f})")

# For live speech: Uncomment below
# recognize_from_speech()

# Plot feature example (for visualization)
y, sr = librosa.load('sample.wav', sr=22050)  # Load your file
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC Spectrogram')
plt.show()