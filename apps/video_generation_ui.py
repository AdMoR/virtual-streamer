#!/usr/bin/env python3
"""
Streamlit UI for Video Generation

A user-friendly interface to submit video generation jobs and monitor their progress.
"""

import streamlit as st
import requests
import time
import json
from typing import Optional, Dict, Any
import os
from virtual_streamer.utils.minio_client import MinIOClient, run_async_in_threads

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

# Page config
st.set_page_config(
    page_title="Virtual Streamer - Video Generation", page_icon="🎬", layout="wide"
)


bucket = "virtual_streamer"
client = MinIOClient(
    access_key=os.environ.get("MINIO_ROOT_USER"),
    secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
    endpoint="http://localhost:9000",
    bucket=bucket
)


def submit_job(request_data: Dict[str, Any]) -> Optional[str]:
    """Submit a video generation job."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/video-generation/generate", json=request_data, timeout=10
        )
        response.raise_for_status()
        result = response.json()
        return result["job_id"]
    except Exception as e:
        st.error(f"Failed to submit job: {e}")
        return None


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the status of a job."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/video-generation/jobs/{job_id}", timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get job status: {e}")
        return None


def list_jobs(limit: int = 20) -> Optional[list]:
    """List recent jobs."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/video-generation/jobs", params={"limit": limit}, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to list jobs: {e}")
        return None


def list_characters() -> Optional[list]:
    """List available characters."""
    try:
        response = requests.get(f"{API_BASE_URL}/characters", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Could not load characters: {e}")
        return []


def list_story_templates() -> Optional[list]:
    """List available story templates."""
    try:
        response = requests.get(f"{API_BASE_URL}/story-templates", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Could not load story templates: {e}")
        return []


def download_video_from_url(url: str) -> Optional[bytes]:
    """Download video content from a presigned URL."""
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None


# Title and description
st.title("🎬 Virtual Streamer - Video Generation")
st.markdown("""
Generate videos from stories using AI! Provide a title for automatic story generation,
or paste your own story text. The system will create a fully produced video with 
matching visuals, voice narration, and subtitles.
""")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Load available characters
    characters = list_characters()
    character_names = ["None (default voice)"] + [c["name"] for c in (characters or [])]

    character_name = st.selectbox(
        "Character Voice", character_names, help="Select a character for voice cloning"
    )
    if character_name == "None (default voice)":
        character_name = None

    with st.expander("🤖 LLM Settings"):
        llm_provider = st.selectbox(
            "Provider", ["anthropic", "openai", "litellm"], index=0
        )
        llm_model = st.text_input("Model", value="claude-sonnet-4-5-20250929")

    with st.expander("🎤 TTS Settings"):
        tts_provider = st.selectbox("Provider", ["fish", "solero", "coqui"], index=0)
        tts_host = st.text_input("Host", value="127.0.0.1")
        tts_port = st.number_input("Port", value=8003, min_value=1, max_value=65535)

    with st.expander("🎧 STT Settings"):
        stt_provider = st.selectbox("Provider", ["whisper", "faster-whisper"], index=0)
        stt_model = st.selectbox(
            "Model", ["tiny", "base", "small", "medium", "large"], index=1
        )

    with st.expander("⚡ Advanced"):
        max_parallel_llm_calls = st.slider(
            "Max Parallel LLM Calls",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of parallel LLM API calls for efficiency",
        )
        output_dir = st.text_input(
            "Output Directory",
            value="./output",
            help="Directory to save generated videos",
        )
        verbose = st.checkbox("Verbose Output", value=False)
        enable_blueprint_dump = st.checkbox(
            "Enable Blueprint Dump",
            value=False,
            help="Upload debug artifacts (TTS audio, videos, subtitles) to MinIO storage for debugging",
        )

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Create New Video", "📊 Job Status", "📋 Recent Jobs"])

with tab1:
    st.header("Create New Video")

    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Title (Generate Story)", "From Story Template", "Paste Story Text"],
        horizontal=True,
    )

    request_data = {
        "character_name": character_name,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "tts_provider": tts_provider,
        "tts_host": tts_host,
        "tts_port": tts_port,
        "stt_provider": stt_provider,
        "stt_model": stt_model,
        "output_dir": output_dir,
        "max_parallel_llm_calls": max_parallel_llm_calls,
        "verbose": verbose,
        "enable_blueprint_dump": enable_blueprint_dump,
    }

    if input_method == "Title (Generate Story)":
        st.markdown("**Enter a title and the AI will generate a story:**")
        title = st.text_input(
            "Video Title",
            placeholder="e.g., 'Fred se lance dans l'IA'",
            help="A short title describing the topic of the video",
        )

        if st.button("🚀 Generate Video", type="primary", disabled=not title):
            request_data["title"] = title
            with st.spinner("Submitting job..."):
                job_id = submit_job(request_data)
                if job_id:
                    st.success(f"✓ Job submitted successfully!")
                    st.info(f"Job ID: `{job_id}`")
                    st.markdown("Switch to the **Job Status** tab to monitor progress.")
                    # Store in session state
                    st.session_state["current_job_id"] = job_id

    elif input_method == "From Story Template":
        st.markdown("**Select a story template and provide a scenario title:**")

        # Load available story templates
        templates = list_story_templates()

        if not templates:
            st.warning("No story templates available. Create templates via the API first.")
        else:
            # Build template options with character info
            template_options = {}
            for t in templates:
                chars = ", ".join(t.get("character_ids", [])) or "No characters"
                label = f"{t['name']} ({chars})"
                template_options[label] = t

            selected_label = st.selectbox(
                "Story Template",
                options=list(template_options.keys()),
                help="Select a predefined story template with its associated characters",
            )
            selected_template = template_options[selected_label]

            # Show template details
            with st.expander("Template Details"):
                st.markdown(f"**Template ID:** `{selected_template['template_id']}`")
                st.markdown(f"**Target Lines:** {selected_template.get('target_lines', 6)}")
                st.markdown(f"**Characters:** {', '.join(selected_template.get('character_ids', [])) or 'None'}")
                st.text_area(
                    "Prompt Preview",
                    value=selected_template.get("prompt", "")[:500] + "..." if len(selected_template.get("prompt", "")) > 500 else selected_template.get("prompt", ""),
                    height=150,
                    disabled=True,
                )

            # Title input for the scenario
            template_title = st.text_input(
                "Scenario Title",
                placeholder="e.g., 'Fred découvre les NFT'",
                help="The specific scenario/topic for this story template",
            )

            if st.button("🚀 Generate Video", type="primary", disabled=not template_title, key="template_generate"):
                request_data["title"] = template_title
                request_data["story_template_id"] = selected_template["template_id"]
                with st.spinner("Submitting job..."):
                    job_id = submit_job(request_data)
                    if job_id:
                        st.success(f"✓ Job submitted successfully!")
                        st.info(f"Job ID: `{job_id}`")
                        st.markdown(f"Using template: **{selected_template['name']}**")
                        st.markdown("Switch to the **Job Status** tab to monitor progress.")
                        st.session_state["current_job_id"] = job_id

    elif input_method == "Paste Story Text":
        st.markdown("**Paste your story text:**")
        story_text = st.text_area(
            "Story Text",
            height=300,
            placeholder="Paste your story here...",
            help="The story dialog to convert into a video",
        )

        if st.button("🚀 Generate Video", type="primary", disabled=not story_text):
            request_data["story_text"] = story_text
            with st.spinner("Submitting job..."):
                job_id = submit_job(request_data)
                if job_id:
                    st.success(f"✓ Job submitted successfully!")
                    st.info(f"Job ID: `{job_id}`")
                    st.markdown("Switch to the **Job Status** tab to monitor progress.")
                    st.session_state["current_job_id"] = job_id

with tab2:
    st.header("Job Status Monitor")

    # Job ID input
    col1, col2 = st.columns([3, 1])
    with col1:
        default_job_id = st.session_state.get("current_job_id", "")
        job_id_input = st.text_input(
            "Job ID", value=default_job_id, placeholder="Enter job ID to monitor"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    if job_id_input:
        # Create placeholder for dynamic updates
        status_container = st.container()

        if auto_refresh:
            # Auto-refresh every 5 seconds
            refresh_interval = 5

            while True:
                with status_container:
                    status = get_job_status(job_id_input)

                    if status:
                        # Status badge
                        status_emoji = {
                            "pending": "⏳",
                            "running": "🔄",
                            "completed": "✅",
                            "failed": "❌",
                        }
                        st.markdown(
                            f"### Status: {status_emoji.get(status['status'], '❓')} {status['status'].upper()}"
                        )

                        # Progress
                        if status.get("progress"):
                            st.info(f"**Progress:** {status['progress']}")

                        # Timestamps
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text(f"Created: {status['created_at']}")

                        # Results or error
                        if status["status"] == "completed" and status.get("result"):
                            st.success("✓ Video generation completed successfully!")
                            result = status["result"]
                            metadata = result.get("metadata", {})

                            # Display video if URL is available
                            video_url = metadata.get("video_url")
                            if video_url:
                                st.markdown("### 📹 Generated Video")
                                # Download and display video
                                local_path = f"/tmp/{os.path.basename(video_url)}"
                                bucket_path = video_url.split("/")[1].lstrip("/")
                                run_async_in_threads(client.download_file(bucket_path, local_path))
                                if local_path:
                                    st.video(local_path)
                                else:
                                    st.warning("Could not load video preview. URL may have expired.")
                                    st.code(f"Video URL: {video_url}")

                            st.markdown("### 📋 Details")
                            with st.expander("Full Result Data"):
                                st.json(result)

                            if result.get("video_path"):
                                st.code(f"Video path: {result['video_path']}")

                            if result.get("config_dump_path"):
                                st.code(f"Config dump: {result['config_dump_path']}")

                            if metadata.get("total_duration"):
                                st.info(f"Video duration: {metadata['total_duration']:.1f} seconds")

                            break  # Stop auto-refresh

                        elif status["status"] == "failed":
                            st.error("❌ Video generation failed")
                            if status.get("error"):
                                st.code(status["error"])
                            break  # Stop auto-refresh

                        # Continue refreshing for pending/running
                        time.sleep(refresh_interval)
                        st.rerun()
                    else:
                        break
        else:
            # Manual refresh
            if st.button("🔄 Refresh Status"):
                st.rerun()

            status = get_job_status(job_id_input)
            if status:
                # Display status (same as above but without auto-refresh)
                status_emoji = {
                    "pending": "⏳",
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                }
                st.markdown(
                    f"### Status: {status_emoji.get(status['status'], '❓')} {status['status'].upper()}"
                )

                if status.get("progress"):
                    st.info(f"**Progress:** {status['progress']}")

                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"Created: {status['created_at']}")

                if status["status"] == "completed" and status.get("result"):
                    st.success("✓ Video generation completed!")
                    result = status["result"]
                    metadata = result.get("metadata", {})

                    # Display video if URL is available
                    video_url = metadata.get("video_url")
                    if video_url:
                        st.markdown("### 📹 Generated Video")
                        video_data = download_video_from_url(video_url)
                        if video_data:
                            st.video(video_data)
                            st.download_button(
                                label="⬇️ Download Video",
                                data=video_data,
                                file_name=f"video_{job_id_input[:8]}.mp4",
                                mime="video/mp4",
                                key="manual_download",
                            )
                        else:
                            st.warning("Could not load video preview.")
                            st.code(f"Video URL: {video_url}")

                    with st.expander("Full Result Data"):
                        st.json(result)

                    if metadata.get("total_duration"):
                        st.info(f"Video duration: {metadata['total_duration']:.1f} seconds")

                elif status["status"] == "failed":
                    st.error("❌ Failed")
                    if status.get("error"):
                        st.code(status["error"])

with tab3:
    st.header("Recent Jobs")

    col1, col2 = st.columns([3, 1])
    with col1:
        limit = st.slider("Number of jobs", min_value=5, max_value=50, value=20)
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    jobs = list_jobs(limit)

    if jobs:
        for job in jobs:
            with st.expander(
                f"{job['status']} - {job['job_id'][:8]}... - {job['created_at']}"
            ):
                st.json(job)

                if st.button(f"Monitor", key=f"monitor_{job['job_id']}"):
                    st.session_state["current_job_id"] = job["job_id"]
                    st.rerun()
    else:
        st.info("No jobs found.")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <small>Virtual Streamer Video Generation UI v1.0.0</small><br>
    <small>API Base URL: {}</small>
</div>
""".format(API_BASE_URL),
    unsafe_allow_html=True,
)
