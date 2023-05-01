import io
import os
import time
from json import JSONDecodeError

import requests
import soundfile as sf
import streamlit as st

if os.environ.get("DOCKER"):
    backend = "http://api:8000"
else:
    backend = "http://0.0.0.0:8000"

INSTRUMENTS = {
    "tru": "Trumpet",
    "sax": "Saxophone",
    "vio": "Violin",
    "gac": "Acoustic Guitar",
    "org": "Organ",
    "cla": "Clarinet",
    "flu": "Flute",
    "voi": "Voice",
    "gel": "Electric Guitar",
    "cel": "Cello",
    "pia": "Piano",
}


def load_audio():
    audio_file = st.file_uploader(label="Upload audio file", type="wav", accept_multiple_files=True)
    if len(audio_file) > 0:
        st.audio(audio_file[0])
        return audio_file
    else:
        return None


def check_for_api(max_tries: int):
    trial_count = 0

    with st.spinner("Waiting for API..."):
        while trial_count <= max_tries:
            try:
                response = health_check()
                if response:
                    return True
            except requests.exceptions.ConnectionError:
                trial_count += 1
                # Handle connection error, e.g. API not yet running
                time.sleep(5)  # Sleep for 1 second before retrying
        st.error("API is not running. Please refresh the page to try again.", icon="🚨")
        st.stop()


def cut_audio_file(audio_file):
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
    audio_file = io.BytesIO()
    sf.write(audio_file, cut_audio_data, sample_rate, format="wav")

    # Display cut audio
    st.audio(audio_file, format="audio/wav")

    return audio_file


def display_predictions(predictions):
    # Display the results using instrument names instead of codes
    for filename, instruments in predictions.items():
        st.subheader(filename)

        if isinstance(instruments, str):
            st.write(instruments)

        else:
            with st.container():
                col1, col2 = st.columns([1, 3])
                present_instruments = [
                    INSTRUMENTS[instrument_code] for instrument_code, presence in instruments.items() if presence
                ]
                if present_instruments:
                    for instrument_name in present_instruments:
                        with col1:
                            st.write(instrument_name)
                        with col2:
                            st.write("✔️")
                else:
                    st.write("No instruments found in this file.")


def health_check():
    # Send a health check request to the API
    response = requests.get(f"{backend}/health-check", timeout=100)

    # Check if the API is running
    if response.status_code == 200:
        return True
    else:
        return False


def predict(file_name, data, model_name):
    file = {"file": data}
    request_data = {"file_name": file_name, "model_name": model_name}

    response = requests.post(
        f"{backend}/predict", params=request_data, files=file, timeout=100
    )  # Replace with your API endpoint URL

    return response


@st.cache_data(show_spinner=False)
def predict_single(audio_file, name, selected_model):
    predictions = {}

    with st.spinner("Predicting instruments..."):
        response = predict(name, audio_file, selected_model)

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            predictions[name] = prediction.get(name, "Error making prediction")
        else:
            st.write(response)
            try:
                st.json(response.json())
            except JSONDecodeError as e:
                st.error(response.text)
            st.stop()
    return predictions


@st.cache_data(show_spinner=False)
def predict_multiple(audio_files, selected_model):
    predictions = {}
    progress_text = "Getting predictions for all files. Please wait."
    progress_bar = st.empty()
    progress_bar.progress(0, text=progress_text)

    num_files = len(audio_files)

    for i, file in enumerate(audio_files):
        name = file.name
        response = predict(name, file, selected_model)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            predictions[name] = prediction[name]
            progress_bar.progress((i + 1) / num_files, text=progress_text)
        else:
            predictions[name] = "Error making prediction."
    progress_bar.empty()
    return predictions


if __name__ == "__main__":
    pass
