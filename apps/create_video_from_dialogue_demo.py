import streamlit as st
import requests
import os
import time
import httpx
from virtual_streamer.utils.utils import speech_to_text_call, get_rmq_channel, ChatQuestion, \
    VideoResponse, SubtitleMode, s3_download
from virtual_streamer.video_server.models import Character
from virtual_streamer.video_server.utils import get_character_data_sync

# --- Configuration ---
# Get the backend URL from environment variable or use a default
BACKEND_URL = os.environ.get("BACKEND_WEBSERVICE_URL", "http://localhost:8000")
PROCESS_ENDPOINT = f"{BACKEND_URL}/process"
ENTITY_URL = os.environ.get("BACKEND_WEBSERVICE_URL", "http://localhost:8002")

# --- Helper Functions ---

def call_process_endpoint(question_text, character_name, subtitle_mode, gpt_response_text, user_name="StreamlitUser"):
    """Sends the request to the backend webservice."""
    payload = {
        "question": {
            "question": question_text,
            "character_name": character_name,
            "subtitle_mode": subtitle_mode,
            "name": user_name
        },
        "gpt_response": gpt_response_text
    }
    try:
        character: Character = get_character_data_sync(character_name)
    except httpx.HTTPStatusError:
        st.error("Invalid character name")
        return None

    try:
        response = requests.post(PROCESS_ENDPOINT, json=payload, timeout=360) # Increased timeout for potentially long process
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="GenAI Video Builder", layout="wide")
st.title("🎬 GenAI Video Builder")
tab1, tab2 = st.tabs(["Generate Video", "Create Character"])

# Import the characters from the backend configuration module
# Ensure this path is correct relative to where you run streamlit
# or that the virtual_streamer package is installed/in PYTHONPATH
try:
    from virtual_streamer.workflows.character_setup import CHARACTERS
    available_characters = list(CHARACTERS.keys())
except ImportError:
    st.error("Could not import character list from backend configuration. Using default.")
    available_characters = ["Jesus", "Jamie", "de"] # Fallback list - ADJUST AS NEEDED

st.sidebar.header("Configuration")
character_name = st.sidebar.selectbox("Select Character:", available_characters)

subtitle_options = ["NONE", "QUESTION", "VOICE_SUBTITLE"]
subtitle_mode = st.sidebar.selectbox("Select Subtitle Mode:", subtitle_options)

st.header("Input")
question = st.text_input("Enter your question for the character:")
# The backend /process endpoint requires the response text, so we need an input for it.
gpt_response = st.text_area("Enter the desired response text for the character:", height=150)

generate_button = st.button("Generate Video")

st.header("Output")
video_placeholder = st.empty()
status_placeholder = st.empty()

if generate_button:
    if not question:
        st.warning("Please enter a question.")
    elif not gpt_response:
        st.warning("Please enter the desired response text.")
    elif not character_name:
        st.warning("Please select a character.")
    else:
        video_placeholder.empty() # Clear previous video
        status_placeholder.info("🚀 Sending request to backend... Please wait.")
        start_time = time.time()

        result = call_process_endpoint(question, character_name, subtitle_mode, gpt_response)

        end_time = time.time()
        processing_time = end_time - start_time

        if result:
            video_path = result.get("video_path")
            s3_path = result.get("s3_path")
            response_text = result.get("response_text") # Might be useful to display

            status_placeholder.success(f"✅ Video generated successfully in {processing_time:.2f} seconds!")
            st.write(f"**Backend Response Text:** {response_text}")
            if s3_path:
                st.write(f"**S3 Path:** {s3_path}")
            else:
                 st.write(f"**Local Server Path:** {video_path}")

            # Display the video - Streamlit needs the file bytes
            try:
                # WARNING: This assumes the frontend can directly access the video_path
                # returned by the backend. This ONLY works if both frontend and backend
                # run on the same machine OR if the video_path is accessible via a
                # shared volume or network path.
                # If backend runs in Docker and frontend outside, this direct access WILL FAIL.
                # A better approach is needed for distributed setups (e.g., backend returns
                # a URL or streams the file).
                # For local testing where both run on host or share volumes, this might work.
                video_path = s3_download(s3_path)
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    video_placeholder.video(video_bytes)
                st.caption(f"Showing video from: {video_path}")
            except FileNotFoundError:
                 st.error(f"❌ Frontend could not find the video file at the path returned by the backend: {video_path}. Ensure the path is accessible.")
            except Exception as e:
                 st.error(f"❌ An error occurred trying to display the video: {e}")

        else:
            # Error message already shown by call_process_endpoint
            status_placeholder.error("❌ Video generation failed.")

# --- Optional: Add instructions or info ---
with tab2:
    st.header("Create New Character")
    name = st.text_input("Name", key="new_char_name")
    description = st.text_area("Description", key="new_char_desc")
    voice_samples = st.text_area("Voice Samples (JSON)", "[]", key="new_char_vs")
    tts_config = st.text_area("TTS Model Config (JSON)", "", key="new_char_tts")
    video_file = st.file_uploader("Representative Video", type=["mp4", "mov", "avi"], key="new_char_video")
    if st.button("Create Character", key="create_char_btn"):
        if not name:
            st.warning("Character name is required.")
        else:
            data = {
                "name": name,
                "description": description,
                "voice_samples": voice_samples,
                "tts_model_config": tts_config
            }
            files = {"video_file": (video_file.name, video_file.getvalue(), video_file.type)} if video_file else {}
            try:
                resp = requests.post(f"{ENTITY_URL}/characters", data=data, files=files, timeout=60)
                resp.raise_for_status()
                st.success(f"Character '{name}' created successfully!")
                # Optionally reload or update available_characters list here
            except requests.RequestException as e:
                st.error(f"Failed to create character: {e}")
    st.markdown("---")
st.sidebar.markdown("---")
st.sidebar.info(f"Backend Service URL: {BACKEND_URL}")
st.sidebar.markdown("""
**Instructions:**
1. Select a character.
2. Choose a subtitle mode.
3. Enter the question you want the character to answer.
4. Enter the exact response text the character should say.
5. Click 'Generate Video'.
""")
