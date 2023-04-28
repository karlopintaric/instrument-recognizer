import json
import time
import soundfile as sf
import io

import requests
import streamlit as st
from ui_backend import health_check, predict


def load_audio():
    audio_file = st.file_uploader(label="Upload audio file", type="wav", accept_multiple_files=True)
    if len(audio_file)> 0:
        st.audio(audio_file[0])
        return audio_file
    else:
        return None

@st.cache_data(show_spinner=False)
def predict_multiple(audio_file):
    predictions = {}
    progress_text = "Getting predictions for all files. Please wait."
    progress_bar = st.empty()
    progress_bar.progress(0, text=progress_text)
    num_files = len(audio_file)
    for i,file in enumerate(audio_file):
        name = file.name
        response = predict(file)
        if response.status_code == 200:
            predictions[name] = response.json()
            #my_bar.progress((i + 1)/num_files, text=progress_text)
            progress_bar.progress((i + 1) / num_files, text=progress_text)
        else:
            predictions[name] = "Error making prediction."
    progress_bar.empty()
    return predictions
    

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
        cut_audio = st.checkbox("✂️ Cut duration", disabled= not predict_valid)    
        
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

    result = st.button("Predict", disabled=not predict_valid)

    if result:
        
        predictions = {}
        if isinstance(audio_file, list):
            predictions = predict_multiple(audio_file)    
        
        else:
            with st.spinner("Predicting instruments..."):
                response = predict(audio_file)
        
            if response.status_code == 200:
                st.success("Audio file uploaded successfully to API.")
                predictions[name] = response.json()
            else:
                st.error("Failed to upload audio file to API.")
                
        # Sort the dictionary alphabetically by key
        sorted_predictions = dict(sorted(predictions.items()))

        # Convert the sorted dictionary to a JSON string
        json_string = json.dumps(sorted_predictions) 
        st.download_button(
            label="Download JSON",
            file_name="predictions.json",
            mime="application/json",
            data=json_string,
        )
                
        st.json(json_string, expanded=True)
    


if __name__ == "__main__":
    main()
