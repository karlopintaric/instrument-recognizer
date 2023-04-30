import json
import time
import soundfile as sf
import io
from json import JSONDecodeError

import requests
import streamlit as st
from ui_backend import health_check, predict, display_predictions, predict_multiple


def load_audio():
    audio_file = st.file_uploader(label="Upload audio file", type="wav", accept_multiple_files=True)
    if len(audio_file)> 0:
        st.audio(audio_file[0])
        return audio_file
    else:
        return None
    

def main():
    st.set_page_config(page_title="Music Instrument Recognition", page_icon="🎸", layout="wide", initial_sidebar_state="collapsed")
    
    st.markdown("<h1 style='text-align: center; color: #FFFFFF; font-size: 3rem;'>Instrument Recognition 🎶</h1>", unsafe_allow_html=True)
    #st.title("Instrument recognizer 🎶")
    selected_model = st.sidebar.selectbox("Select Model", ("Accuracy", "Speed"), help="Select a slower but more accurate model or a faster but less accurate model")
    audio_file = load_audio()

    # Send a health check request to the API in a loop until it is running
    api_running = False
    with st.spinner("Waiting for API..."):
        trial_count = 0
        max_tries = 6
        while not api_running:
            try:
                response = health_check()
                if response:
                    api_running = True
            except requests.exceptions.ConnectionError:
                trial_count += 1
                # Handle connection error, e.g. API not yet running
                if trial_count > max_tries:
                    st.error("API is not running. Please refresh the page to try again.", icon="🚨")
                    st.stop()
                time.sleep(5)  # Sleep for 1 second before retrying

    # Enable or disable a button based on API status
    predict_valid = False
    cut_valid = True
    if api_running:
        st.info("API is running", icon="🤖")

    if audio_file:
        num_files = len(audio_file)
        st.write(f"Number of uploaded files: {num_files}")
        predict_valid = True
        if len(audio_file) > 1:
            cut_valid = False
        else:
            audio_file = audio_file[0]
            name = audio_file.name
        
    if cut_valid:
        cut_audio = st.checkbox("✂️ Cut duration", disabled= not predict_valid, help="Cut a long audio file. Model works best if audio is around 20 seconds")    
        
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
            audio_file = io.BytesIO()
            sf.write(audio_file, cut_audio_data, sample_rate, format='wav')
            #audio_file.seek(0)

            # Display cut audio
            st.audio(audio_file, format='audio/wav')

    result = st.button("Predict", disabled=not predict_valid, help="Send the audio to API to get a prediction")

    if result:
        
        predictions = {}
        if isinstance(audio_file, list):
            predictions = predict_multiple(audio_file, selected_model)    
        
        else:
            with st.spinner("Predicting instruments..."):
                response = predict(name, "wav", audio_file, selected_model)
        
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
                
        # Sort the dictionary alphabetically by key
        sorted_predictions = dict(sorted(predictions.items()))
        
        # Convert the sorted dictionary to a JSON string
        json_string = json.dumps(sorted_predictions) 
        st.download_button(
            label="Download JSON",
            file_name="predictions.json",
            mime="application/json",
            data=json_string,
            help="Download the predictions in JSON format"
        )
                
        display_predictions(sorted_predictions)
    


if __name__ == "__main__":
    main()
