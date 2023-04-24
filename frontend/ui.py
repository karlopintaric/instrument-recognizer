import json
import time
import soundfile as sf
import io

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
    model_selection = st.sidebar.selectbox("Select Model", ("Model 1", "Model 2"))
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
        
    cut_audio = st.checkbox("✂️ Cut duration", disabled= not valid)    
    if cut_audio:
        
        audio_data, sample_rate = sf.read(audio_file)

        # Display audio duration
        duration = len(audio_data) / sample_rate
        st.info(f"Audio Duration: {duration} seconds")

        # Get start and end time for cutting
        start_time = st.number_input("Start Time (seconds)", min_value=0.0, max_value=duration, step=0.1)
        end_time = st.number_input("End Time (seconds)", min_value=start_time, value=duration, max_value=duration, step=0.1)

        # Convert start and end time to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Cut audio
        cut_audio_data = audio_data[start_sample:end_sample]

        # Create a temporary in-memory file for cut audio
        cut_audio_file = io.BytesIO()
        sf.write(cut_audio_file, cut_audio_data, sample_rate, format='wav')
        cut_audio_file.seek(0)

        # Display cut audio
        st.audio(cut_audio_file, format='audio/wav')
        audio_file = cut_audio_file.getvalue()

    result = st.button("Predict", disabled=not valid)

    if result:
        with st.spinner("Predicting instruments..."):
            response = predict(audio_file)
        
        if response.status_code == 200:
            st.success("Audio file uploaded successfully to API.")
            st.write(json.dumps(response.json()))
            st.write(response)
        else:
            st.error("Failed to upload audio file to API.")

if __name__ == "__main__":
    main()
