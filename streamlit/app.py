import io
import streamlit as st
from backend import load_labels, load_model, predict, process_audio
import torchaudio

def load_audio():
    audio_file = st.file_uploader(label="Upload audio file")
    if audio_file is not None:
        audio_data = audio_file.getvalue()
        st.audio(audio_data)
        return io.BytesIO(audio_data)
    else:
        return None

def main():
    st.title("Instrument recognizer")
    
    audio = load_audio()
    labels = load_labels()
    model = load_model()

    result = st.button("Predict")
    if result:
        predict(model, audio, labels)

    
if __name__=="__main__":
    main()