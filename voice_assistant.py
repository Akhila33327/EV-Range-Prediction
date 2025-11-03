import streamlit as st
import sounddevice as sd
import whisper
import scipy.io.wavfile as wav
import tempfile
import pyttsx3
from chatbot_engine import ask_ev_chatbot

@st.cache_resource
def load_stt():
    return whisper.load_model("small")

stt = load_stt()
tts = pyttsx3.init()
tts.setProperty('rate', 165)

def speak(text):
    tts.say(text)
    tts.runAndWait()

def record(seconds=5, fs=16000):
    rec = sd.rec(int(seconds*fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(file.name, fs, rec)
    return file.name

st.title("🎙️ Voice-Based EV Assistant")

sec = st.slider("Record Duration", 2, 10, 5)

if st.button("🎤 Start Talking"):
    audio = record(sec)
    text = stt.transcribe(audio, fp16=False)['text']
    st.write(f"**You:** {text}")

    reply = ask_ev_chatbot(text)
    st.write(f"**EV Assistant:** {reply}")

    speak(reply)
