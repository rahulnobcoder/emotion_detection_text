import streamlit as st
import soundfile as sf
from io import BytesIO
from tensorflow.keras.models import load_model
from functions import *
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr

r = sr.Recognizer()

emotion_dict = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

with open(r'models\tokenizer.json', 'r') as json_file:
    tokenizer_json = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
model = load_model(r'models\final_model.keras')

# Set the title of the Streamlit app
st.title("Emotion detection using text")

# Text input for some data
input_text = st.text_input("Enter some text:")

# File uploader for audio files
audio_file = st.file_uploader("Upload an audio file:", type=["mp3", "wav"])

# Button to upload
if st.button("Upload"):
    if audio_file:
        # Read the audio file
        audio_data, samplerate = sf.read(audio_file)
        
        # Convert the audio file to WAV format and save it
        output_file_path = 'uploaded_audio.wav'
        sf.write(output_file_path, audio_data, samplerate)
        
        st.audio(audio_file)
    else:
        st.error("Please upload an audio file.")

# Button to predict

if st.button("Predict"):
    if input_text:

        proc_text=process_sentence(input_text)
        sequences_test = tokenizer.texts_to_sequences([proc_text])
        sentence = pad_sequences(sequences_test, maxlen=229, truncating='pre')
        result = np.argmax(model.predict(sentence), axis=1)
        st.write(f"pred: {emotion_dict[result[0]]}")
    elif audio_file:
        file='uploaded_audio.wav'
        with sr.AudioFile(file) as source:
    # Record the audio data
            audio_data = r.record(source)
            
            try:
                text = r.recognize_google(audio_data)
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error("Error accessing Google Speech Recognition service; {0}".format(e))
        st.write(f"text_recognised: {text}")
        proc_text=process_sentence(text)
        sequences_test = tokenizer.texts_to_sequences([proc_text])
        sentence = pad_sequences(sequences_test, maxlen=229, truncating='pre')
        result = np.argmax(model.predict(sentence), axis=1)
        st.write(f"pred: {emotion_dict[result[0]]}")
    else:
        st.error("Please enter some text.")

# Additional message at the bottom of the page
st.write("Thank you for using the app!")
