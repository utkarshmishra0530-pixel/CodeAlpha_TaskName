import speech_recognition as sr
from textblob import TextBlob

def analyze_emotion(text):
    # Use TextBlob for sentiment analysis (polarity: -1 to 1, subjectivity: 0 to 1)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Map to emotions (simplified heuristic)
    if polarity > 0.5:
        emotion = "Happy/Excitement"
    elif polarity < -0.5:
        emotion = "Sad/Anger"
    elif subjectivity > 0.5:
        emotion = "Fear/Anxiety"  # High subjectivity often indicates uncertainty
    else:
        emotion = "Neutral"
    
    return emotion, polarity, subjectivity

def recognize_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Transcribed: {text}")
            emotion, pol, subj = analyze_emotion(text)
            print(f"Detected Emotion: {emotion} (Polarity: {pol:.2f}, Subjectivity: {subj:.2f})")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Speech service error")

# Example with text input
text_input = "I'm so excited about this new project!"  # Replace with your own
emotion, pol, subj = analyze_emotion(text_input)
print(f"Text: {text_input}")
print(f"Emotion: {emotion} (Polarity: {pol:.2f}, Subjectivity: {subj:.2f})")

# Uncomment to test with speech (requires microphone)
# recognize_from_speech()