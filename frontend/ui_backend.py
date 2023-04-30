import requests
import os
import streamlit as st

INSTRUMENTS = {
    'tru': 'Trumpet',
    'sax': 'Saxophone',
    'vio': 'Violin',
    'gac': 'Acoustic Guitar',
    'org': 'Organ',
    'cla': 'Clarinet',
    'flu': 'Flute',
    'voi': 'Voice',
    'gel': 'Electric Guitar',
    'cel': 'Cello',
    'pia': 'Piano'
}

def display_predictions(predictions):

    # Display the results using instrument names instead of codes
    for filename, instruments in predictions.items():
        st.subheader(filename)
        
        if isinstance(instruments, str):
            st.write(instruments)
        
        else:
            with st.container():
                col1, col2 = st.columns([1, 3])
                present_instruments = [INSTRUMENTS[instrument_code] for instrument_code, presence in instruments.items() if presence]
                if present_instruments:
                    for instrument_name in present_instruments:
                        with col1:
                            st.write(instrument_name)
                        with col2:
                            st.write("✔️")
                else:
                    st.write("No instruments found in this file.")                        
        
if os.environ.get("DOCKER"):
    backend = "http://api:8000"
else:
    backend = "http://0.0.0.0:8000"


def health_check():
    # Send a health check request to the API
    response = requests.get(f"{backend}/health-check", timeout=100)

    # Check if the API is running
    if response.status_code == 200:
        return True
    else:
        return False

def predict(file_name, file_type, data, model_name):
    file = {"file": data}
    request_data = {"file_name": file_name, "file_type": file_type, "model_name": model_name}
    #files = {"file": file.getvalue()}
    #data = {"audio": audio_data, "model": model}

    response = requests.post(f"{backend}/predict", params=request_data, files=file, timeout=100)  # Replace with your API endpoint URL

    return response

@st.cache_data(show_spinner=False)
def predict_multiple(audio_files, selected_model):
    predictions = {}
    progress_text = "Getting predictions for all files. Please wait."
    progress_bar = st.empty()
    progress_bar.progress(0, text=progress_text)
    num_files = len(audio_files)
    for i,file in enumerate(audio_files):
        name = file.name
        response = predict(name, "wav", file, selected_model)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            predictions[name] = prediction[name]
            #my_bar.progress((i + 1)/num_files, text=progress_text)
            progress_bar.progress((i + 1) / num_files, text=progress_text)
        else:
            predictions[name] = "Error making prediction."
    progress_bar.empty()
    return predictions


if __name__ == "__main__":
    pass
