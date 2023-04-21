import json
import time

import requests
import streamlit as st
from ui_backend import health_check, predict


def load_audio():
    audio_file = st.file_uploader(label="Upload audio file", type="wav")
    if audio_file is not None:
        st.audio(audio_file)
        return audio_file
    else:
        return None


def main():
    st.title("Instrument recognizer")
    audio_file = load_audio()

    # Send a health check request to the API in a loop until it is running
    api_running = False
    with st.spinner("Waiting for API..."):
        while not api_running:
            try:
                response = health_check()
                if response:
                    api_running = True
            except requests.exceptions.ConnectionError:
                # Handle connection error, e.g. API not yet running
                time.sleep(5)  # Sleep for 1 second before retrying

    # Enable or disable a button based on API status
    valid = False
    if api_running:
        st.info("API is running", icon="🤖")

    if audio_file:
        valid = True

    result = st.button("Predict", disabled=not valid)

    if result:
        with st.spinner("Predicting instruments..."):
            prediction = predict(audio_file)
        st.write(json.dumps(prediction.json()))
        st.write(prediction)


if __name__ == "__main__":
    main()
