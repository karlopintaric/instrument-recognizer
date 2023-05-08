import io
import os
import time
from json import JSONDecodeError

import requests
import soundfile as sf
import streamlit as st

if os.environ.get("IS_DOCKER", False):
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
    """
    Upload a WAV audio file and display it in a Streamlit app.

    :return: A BytesIO object representing the uploaded audio file, or None if no file was uploaded.
    :rtype: Optional[BytesIO]
    """

    audio_file = st.file_uploader(label="Upload audio file", type="wav", accept_multiple_files=True)
    if len(audio_file) > 0:
        st.audio(audio_file[0])
        return audio_file
    else:
        return None


@st.cache_data(show_spinner=False)
def check_for_api(max_tries: int):
    """
    Check if the API is running by making a health check request.

    :param max_tries: The maximum number of attempts to check the API's health.
    :type max_tries: int
    :return: True if the API is running, False otherwise.
    :rtype: bool
    """
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


def cut_audio_file(audio_file, name):
    """
    Cut an audio file and return the cut audio data as a tuple.

    :param audio_file: The path of the audio file to be cut.
    :type audio_file: str
    :param name: The name of the audio file to be cut.
    :type name: str
    :raises RuntimeError: If the audio file cannot be read.
    :return: A tuple containing the name and the cut audio data as a BytesIO object.
    :rtype: tuple
    """
    try:
        audio_data, sample_rate = sf.read(audio_file)
    except RuntimeError as e:
        raise e

    # Display audio duration
    duration = round(len(audio_data) / sample_rate, 2)
    st.info(f"Audio Duration: {duration} seconds")

    # Get start and end time for cutting
    start_time = st.number_input("Start Time (seconds)", min_value=0.0, max_value=duration - 1, step=0.1)
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
    audio_file = (name, audio_file)

    return audio_file


def display_predictions(predictions: dict):
    """
    Display the predictions using instrument names instead of codes.

    :param predictions: A dictionary containing the filenames and instruments detected in them.
    :type predictions: dict
    """

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
    """
    Sends a health check request to the API and checks if it's running.

    :return: Returns True if the API is running, else False.
    :rtype: bool
    """

    # Send a health check request to the API
    response = requests.get(f"{backend}/health-check", timeout=100)

    # Check if the API is running
    if response.status_code == 200:
        return True
    else:
        return False


def predict(data, model_name):
    """
    Sends a POST request to the API with the provided data and model name.

    :param data: The audio data to be used for prediction.
    :type data: bytes
    :param model_name: The name of the model to be used for prediction.
    :type model_name: str
    :return: The response from the API.
    :rtype: requests.Response
    """

    file = {"file": data}
    request_data = {"model_name": model_name}

    response = requests.post(
        f"{backend}/predict", params=request_data, files=file, timeout=100
    )  # Replace with your API endpoint URL

    return response


@st.cache_data(show_spinner=False)
def predict_single(audio_file, name, selected_model):
    """
    Predicts the instruments in a single audio file using the selected model.

    :param audio_file: The audio file to be used for prediction.
    :type audio_file: bytes
    :param name: The name of the audio file.
    :type name: str
    :param selected_model: The name of the selected model.
    :type selected_model: str
    :return: A dictionary containing the predicted instruments for the audio file.
    :rtype: dict
    """

    predictions = {}

    with st.spinner("Predicting instruments..."):
        response = predict(audio_file, selected_model)

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            predictions[name] = prediction.get(name, "Error making prediction")
        else:
            st.write(response)
            try:
                st.json(response.json())
            except JSONDecodeError:
                st.error(response.text)
            st.stop()
    return predictions


@st.cache_data(show_spinner=False)
def predict_multiple(audio_files, selected_model):
    """
    Generates predictions for multiple audio files using the selected model.

    :param audio_files: A list of audio files to make predictions on.
    :type audio_files: List[UploadedFile]
    :param selected_model: The model to use for making predictions.
    :type selected_model: str
    :return: A dictionary where the keys are the names of the audio files and the values are the predicted labels.
    :rtype: Dict[str, str]
    """

    predictions = {}
    progress_text = "Getting predictions for all files. Please wait."
    progress_bar = st.empty()
    progress_bar.progress(0, text=progress_text)

    num_files = len(audio_files)

    for i, file in enumerate(audio_files):
        name = file.name
        response = predict(file, selected_model)
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
