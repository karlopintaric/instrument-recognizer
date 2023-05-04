import os,sys
from pathlib import Path
import json

import streamlit as st

from ui_backend import (
    check_for_api,
    cut_audio_file,
    display_predictions,
    load_audio,
    predict_multiple,
    predict_single,
)


def main():
    # Page settings
    st.set_page_config(
        page_title="Music Instrument Recognition", page_icon="🎸", layout="wide", initial_sidebar_state="collapsed"
    )

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        selected_model = st.selectbox(
            "Select Model",
            ("Accuracy", "Speed"),
            help="Select a slower but more accurate model or a faster but less accurate model",
        )

    # Main title
    st.markdown(
        "<h1 style='text-align: center; color: #FFFFFF; font-size: 3rem;'>Instrument Recognition 🎶</h1>",
        unsafe_allow_html=True,
    )

    # Upload widget
    audio_file = load_audio()

    # Send a health check request to the API in a loop until it is running
    api_running = check_for_api(6)

    # Enable or disable a button based on API status
    predict_valid = False
    cut_valid = False

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
            cut_valid=True

    if cut_valid:
        cut_audio = st.checkbox(
            "✂️ Cut duration",
            disabled=not predict_valid,
            help="Cut a long audio file. Model works best if audio is around 20 seconds",
        )

        if cut_audio:
            audio_file = cut_audio_file(audio_file)
            

    result = st.button("Predict", disabled=not predict_valid, help="Send the audio to API to get a prediction")

    if result:
        predictions = {}
        if isinstance(audio_file, list):
            predictions = predict_multiple(audio_file, selected_model)

        else:
            predictions = predict_single(audio_file, selected_model)
            

        # Sort the dictionary alphabetically by key
        sorted_predictions = dict(sorted(predictions.items()))

        # Convert the sorted dictionary to a JSON string
        json_string = json.dumps(sorted_predictions)
        st.download_button(
            label="Download JSON",
            file_name="predictions.json",
            mime="application/json",
            data=json_string,
            help="Download the predictions in JSON format",
        )

        display_predictions(sorted_predictions)


if __name__ == "__main__":
    main()
