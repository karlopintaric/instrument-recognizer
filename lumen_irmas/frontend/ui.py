import streamlit as st
from lumen_irmas.frontend.ui_backend import predict
import json

def load_audio():
    audio_file = st.file_uploader(label="Upload audio file")
    if audio_file is not None:
        st.audio(audio_file)
        #length = audio.size / 16000
        #st.slider(min_value=0.0, max_value=length, value=(0.0, length), step=0.5, label="Duration in seconds")
        return audio_file
    else:
        return None

def main():
    st.title("Instrument recognizer")
    audio_file = load_audio()
    
    result = st.button("Predict")
    if result:
        prediction = predict(audio_file)
        st.write(json.dumps(prediction.json()))

    
if __name__=="__main__":
    main()