import threading
import subprocess
import time

def launch_api():
    # Launch the API script with uvicorn from the api directory
    api_dir = "api"
    uvicorn_command = "uvicorn"
    subprocess.run([uvicorn_command, "app:app", "--host", "0.0.0.0" , "--port", "8000"], cwd=api_dir)

def launch_streamlit():
    # Launch the Streamlit script from the streamlit directory
    streamlit_dir = "frontend"
    streamlit_script = "ui.py"
    subprocess.run(["streamlit", "run", streamlit_script, "--server.port", "8501"], cwd=streamlit_dir)

if __name__ == '__main__':
# Launch Streamlit and API in separate threads
    streamlit_thread = threading.Thread(target=launch_streamlit)
    api_thread = threading.Thread(target=launch_api)
    streamlit_thread.start()
    api_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        print("Keyboard interrupt received, stopping threads...")
        streamlit_thread.join()
        api_thread.join()
        print("Threads stopped.")