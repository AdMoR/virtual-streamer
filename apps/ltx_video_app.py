"""
LTX Video Generator - Streamlit App

A web interface for generating videos using the LTX Video API.

Features:
- Single Prompt Mode: Generate a single video from a text prompt
- Story Mode: Generate a full story video with multiple scenes

Run with:
    streamlit run apps/ltx_video_app.py
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from virtual_streamer.video_generation.ltx_client import (
    LTXVideoClient,
    LTXVideoConfig,
    VideoGenerationParams,
    VideoGenerationResult,
)
from virtual_streamer.video_generation.config import DialogLine, StoryOutput
from virtual_streamer.video_generation.story_to_video import (
    story_to_video,
    StoryVideoResult,
)
from virtual_streamer.video_generation.ltx_prompt_builder import (
    build_ltx_prompt,
    build_prompts_from_story,
)


# Page configuration
st.set_page_config(
    page_title="LTX-2 Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-family: 'SF Mono', 'Fira Code', monospace;
    }
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "generated_videos": [],
        "last_result": None,
        "generation_in_progress": False,
        "progress": 0.0,
        "status_message": "",
        # Story mode state
        "story_output": None,
        "story_video_result": None,
        "story_prompts_preview": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_async(coro):
    """Run an async coroutine in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def generate_video_async(
    config: LTXVideoConfig,
    params: VideoGenerationParams,
    output_dir: str,
    progress_placeholder,
    status_placeholder,
) -> VideoGenerationResult:
    """Generate video with progress updates."""
    
    def progress_callback(progress: float, message: str):
        st.session_state.progress = progress
        st.session_state.status_message = message
        progress_placeholder.progress(progress, text=message)
    
    async with LTXVideoClient(config) as client:
        result = await client.generate_video(
            params=params,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
    
    return result


async def generate_story_video_async(
    story_output: StoryOutput,
    config: LTXVideoConfig,
    video_params: VideoGenerationParams,
    output_dir: str,
    progress_placeholder,
    status_placeholder,
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting.",
) -> StoryVideoResult:
    """Generate video from story with progress updates."""
    
    def progress_callback(current: int, total: int, message: str):
        progress = current / max(total, 1)
        st.session_state.progress = progress
        st.session_state.status_message = message
        progress_placeholder.progress(progress, text=f"[{current}/{total}] {message}")
    
    result = await story_to_video(
        story_output=story_output,
        comfyui_config=config,
        video_params=video_params,
        output_dir=output_dir,
        progress_callback=progress_callback,
        style_suffix=style_suffix,
    )
    
    return result


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Server settings
        st.markdown("### Server")
        server_url = st.text_input(
            "LTX Video API Server URL",
            value="http://localhost:8081",
            help="URL of the LTX Video API server"
        )
        
        # Check server connection
        if st.button("🔌 Test Connection", use_container_width=True):
            try:
                import httpx
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(f"{server_url}/health")
                    if response.status_code == 200:
                        st.success("✅ Connected!")
                    else:
                        st.warning(f"⚠️ Server responded with {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
        
        st.divider()
        
        # Video settings
        st.markdown("### Video Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            width = st.selectbox(
                "Width",
                options=[640, 768, 896, 1024, 1280, 1920],
                index=4,  # Default to 1280
                help="Video width (multiple of 32)"
            )
        with col2:
            height = st.selectbox(
                "Height",
                options=[384, 448, 512, 576, 640, 720, 1080],
                index=5,  # Default to 720
                help="Video height (multiple of 32)"
            )
        
        duration = st.slider(
            "Duration (seconds)",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="Video duration (actual may vary due to frame constraints)"
        )
        
        fps = st.selectbox(
            "Frame Rate (FPS)",
            options=[16, 24, 30],
            index=1,
            help="Frames per second"
        )
        
        st.divider()
        
        # Generation settings
        st.markdown("### Generation")
        
        steps = st.slider(
            "Sampling Steps",
            min_value=10,
            max_value=50,
            value=20,
            help="More steps = higher quality but slower"
        )
        
        cfg_scale = st.slider(
            "CFG Scale",
            min_value=1.0,
            max_value=15.0,
            value=4.0,
            step=0.5,
            help="How closely to follow the prompt"
        )
        
        seed = st.number_input(
            "Seed",
            min_value=-1,
            max_value=2147483647,
            value=-1,
            help="-1 for random seed"
        )
        
        enable_audio = st.checkbox(
            "Generate Audio",
            value=True,
            help="Generate synchronized audio (if supported)"
        )
        
        st.divider()
        
        # Output settings
        st.markdown("### Output")
        output_dir = st.text_input(
            "Output Directory",
            value="./generated_videos",
            help="Where to save generated videos"
        )
        
        return {
            "server_url": server_url,
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "enable_audio": enable_audio,
            "output_dir": output_dir,
        }


def render_main_content(config_values: dict):
    """Render the main content area."""
    
    # Header
    st.markdown("""
    # 🎬 LTX-2 Video Generator
    
    Generate AI videos from text prompts using the LTX-2 model through ComfyUI.
    """)
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["📝 Single Prompt", "📖 Story Mode"])
    
    with tab1:
        render_single_prompt_tab(config_values)
    
    with tab2:
        render_story_mode_tab(config_values)


def render_single_prompt_tab(config_values: dict):
    """Render the single prompt generation tab."""
    
    # Prompt inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "📝 Prompt",
            placeholder="Describe the video you want to generate...\n\nExample: A serene forest at sunrise, golden light filtering through the trees, gentle mist rising from the ground, cinematic quality",
            height=150,
            help="Describe what you want to see in the video"
        )
    
    with col2:
        negative_prompt = st.text_area(
            "🚫 Negative Prompt",
            placeholder="What to avoid...\n\nExample: blurry, low quality, artifacts, distorted",
            height=150,
            help="Describe what you want to avoid"
        )
    
    # Show calculated parameters
    params = VideoGenerationParams(
        prompt=prompt or "placeholder",
        negative_prompt=negative_prompt,
        width=config_values["width"],
        height=config_values["height"],
        duration_seconds=config_values["duration"],
        fps=config_values["fps"],
        steps=config_values["steps"],
        cfg_scale=config_values["cfg_scale"],
        seed=config_values["seed"],
        enable_audio=config_values["enable_audio"],
    )
    
    with st.expander("📊 Video Parameters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resolution", f"{params.width}×{params.height}")
        col2.metric("Frame Count", params.frame_count)
        col3.metric("Actual Duration", f"{params.actual_duration:.2f}s")
        col4.metric("Total Frames", f"{params.frame_count} @ {params.fps}fps")
    
    st.divider()
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Generate Video",
            type="primary",
            use_container_width=True,
            disabled=not prompt or st.session_state.generation_in_progress
        )
    
    # Progress and status area
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if generate_clicked and prompt:
        st.session_state.generation_in_progress = True
        
        try:
            # Create config and params
            config = LTXVideoConfig(
                server_url=config_values["server_url"],
                timeout=600.0  # 10 minute timeout for video generation
            )
            
            params = VideoGenerationParams(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=config_values["width"],
                height=config_values["height"],
                duration_seconds=config_values["duration"],
                fps=config_values["fps"],
                steps=config_values["steps"],
                cfg_scale=config_values["cfg_scale"],
                seed=config_values["seed"],
                enable_audio=config_values["enable_audio"],
            )
            
            # Ensure output directory exists
            output_dir = config_values["output_dir"]
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate video
            with st.spinner("Generating video..."):
                result = run_async(
                    generate_video_async(
                        config=config,
                        params=params,
                        output_dir=output_dir,
                        progress_placeholder=progress_placeholder,
                        status_placeholder=status_placeholder,
                    )
                )
            
            st.session_state.last_result = result
            st.session_state.generated_videos.append({
                "result": result,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            status_placeholder.success(f"✅ Video generated successfully!")
            
        except Exception as e:
            status_placeholder.error(f"❌ Generation failed: {str(e)}")
            st.exception(e)
        
        finally:
            st.session_state.generation_in_progress = False
    
    # Display result
    if st.session_state.last_result:
        st.divider()
        render_video_result(st.session_state.last_result)


def render_story_mode_tab(config_values: dict):
    """Render the story mode generation tab."""
    
    st.markdown("""
    ### Story-to-Video Generation
    
    Create a multi-scene video from a story. Each dialog line becomes a video segment
    with synchronized audio, then all segments are concatenated into a final video.
    """)
    
    # Story input method
    input_method = st.radio(
        "Input Method",
        options=["Manual Dialog Lines", "Generate from Title"],
        horizontal=True,
        help="Choose how to create the story"
    )
    
    story_output = None
    
    if input_method == "Manual Dialog Lines":
        st.markdown("#### Define Dialog Lines")
        st.caption("Add dialog lines with character, text, and scene description.")
        
        # Initialize dialog lines in session state
        if "manual_dialog_lines" not in st.session_state:
            st.session_state.manual_dialog_lines = [
                {"character_id": "narrator", "text": "", "scene_description": ""}
            ]
        
        # Render dialog line inputs
        lines_to_remove = []
        for i, line in enumerate(st.session_state.manual_dialog_lines):
            with st.expander(f"Scene {i+1}", expanded=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    line["character_id"] = st.text_input(
                        "Character",
                        value=line.get("character_id", "narrator"),
                        key=f"char_{i}",
                        help="Character ID (e.g., 'narrator', 'fred', 'jamy')"
                    )
                with col2:
                    line["text"] = st.text_area(
                        "Dialog Text",
                        value=line.get("text", ""),
                        key=f"text_{i}",
                        height=80,
                        help="What the character says (will be used for audio)"
                    )
                line["scene_description"] = st.text_area(
                    "Scene Description",
                    value=line.get("scene_description", ""),
                    key=f"scene_{i}",
                    height=80,
                    help="Visual description of the scene (what should be shown)"
                )
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    lines_to_remove.append(i)
        
        # Remove marked lines
        for i in reversed(lines_to_remove):
            st.session_state.manual_dialog_lines.pop(i)
            st.rerun()
        
        # Add new line button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("➕ Add Scene"):
                st.session_state.manual_dialog_lines.append(
                    {"character_id": "narrator", "text": "", "scene_description": ""}
                )
                st.rerun()
        
        # Build StoryOutput from manual lines
        if st.session_state.manual_dialog_lines:
            valid_lines = [
                DialogLine(
                    character_id=line["character_id"],
                    text=line["text"],
                    scene_description=line["scene_description"]
                )
                for line in st.session_state.manual_dialog_lines
                if line["text"].strip() and line["scene_description"].strip()
            ]
            if valid_lines:
                story_output = StoryOutput(
                    title="Manual Story",
                    story_plan="Manually created dialog lines",
                    dialog=valid_lines
                )
    
    else:  # Generate from Title
        st.markdown("#### Generate Story from Title")
        
        title = st.text_input(
            "Story Title/Topic",
            placeholder="e.g., 'How does artificial intelligence work?'",
            help="The AI will generate a story with dialog lines based on this title"
        )
        
        story_template_id = st.text_input(
            "Story Template ID (optional)",
            placeholder="e.g., 'cest_pas_sorcier'",
            help="ID of a story template to use for generation style"
        )
        
        if st.button("🎭 Generate Story", disabled=not title):
            with st.spinner("Generating story..."):
                try:
                    # Import the story generator
                    from virtual_streamer.api.high_level.video_generation import run_story_generator
                    
                    story_output = run_async(
                        run_story_generator(
                            title=title,
                            story_template_id=story_template_id if story_template_id else None
                        )
                    )
                    st.session_state.story_output = story_output
                    st.success(f"✅ Story generated: {story_output.title}")
                except Exception as e:
                    st.error(f"❌ Story generation failed: {str(e)}")
                    st.exception(e)
        
        # Use stored story output
        if st.session_state.story_output:
            story_output = st.session_state.story_output
    
    # Preview prompts
    if story_output:
        st.divider()
        st.markdown("### 📋 Story Preview")
        
        st.markdown(f"**Title:** {story_output.title}")
        st.markdown(f"**Scenes:** {len(story_output.dialog)}")
        
        # Generate and show prompts
        prompts = build_prompts_from_story(
            story_output,
            style_suffix=config_values.get("style_suffix", "Cinematic quality, smooth motion.")
        )
        
        with st.expander("View LTX-2 Prompts", expanded=False):
            for i, p in enumerate(prompts):
                st.markdown(f"**Scene {i+1}** ({p['character_id']})")
                st.code(p["prompt"], language=None)
                st.caption(f"Dialog: {p['dialog_line'].text[:100]}...")
                st.divider()
        
        # Style suffix input
        style_suffix = st.text_input(
            "Style Suffix",
            value="Cinematic quality, smooth motion, natural lighting.",
            help="Added to each prompt for consistent style"
        )
        
        st.divider()
        
        # Generate video button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_story_clicked = st.button(
                "🎬 Generate Story Video",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.generation_in_progress
            )
        
        # Progress and status area
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if generate_story_clicked:
            st.session_state.generation_in_progress = True
            
            try:
                # Create config and params
                config = LTXVideoConfig(
                    server_url=config_values["server_url"],
                    timeout=600.0
                )
                
                video_params = VideoGenerationParams(
                    prompt="",  # Will be set per segment
                    width=config_values["width"],
                    height=config_values["height"],
                    duration_seconds=config_values["duration"],
                    fps=config_values["fps"],
                    steps=config_values["steps"],
                    cfg_scale=config_values["cfg_scale"],
                    seed=config_values["seed"],
                    enable_audio=config_values["enable_audio"],
                )
                
                # Ensure output directory exists
                output_dir = config_values["output_dir"]
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Generate story video
                with st.spinner("Generating story video..."):
                    result = run_async(
                        generate_story_video_async(
                            story_output=story_output,
                            config=config,
                            video_params=video_params,
                            output_dir=output_dir,
                            progress_placeholder=progress_placeholder,
                            status_placeholder=status_placeholder,
                            style_suffix=style_suffix,
                        )
                    )
                
                st.session_state.story_video_result = result
                status_placeholder.success(f"✅ Story video generated successfully!")
                
            except Exception as e:
                status_placeholder.error(f"❌ Generation failed: {str(e)}")
                st.exception(e)
            
            finally:
                st.session_state.generation_in_progress = False
        
        # Display story video result
        if st.session_state.story_video_result:
            st.divider()
            render_story_video_result(st.session_state.story_video_result)


def render_story_video_result(result: StoryVideoResult):
    """Render the generated story video result."""
    st.markdown("## 🎥 Generated Story Video")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video player
        if Path(result.final_video_path).exists():
            st.video(result.final_video_path)
        else:
            st.warning(f"Video file not found: {result.final_video_path}")
    
    with col2:
        # Video info
        st.markdown("### Video Info")
        st.markdown(f"**Title:** {result.story_title}")
        st.markdown(f"**Duration:** {result.total_duration_seconds:.2f}s")
        st.markdown(f"**Segments:** {len(result.segments)}")
        
        # Download button
        if Path(result.final_video_path).exists():
            with open(result.final_video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Video",
                    data=f,
                    file_name=Path(result.final_video_path).name,
                    mime="video/mp4",
                    use_container_width=True
                )
    
    # Segment details
    with st.expander("📊 Segment Details", expanded=False):
        for seg in result.segments:
            st.markdown(f"**Segment {seg.index + 1}** ({seg.dialog_line.character_id})")
            st.caption(f"Duration: {seg.duration_seconds:.2f}s")
            st.caption(f"Dialog: {seg.dialog_line.text[:100]}...")
            if Path(seg.video_path).exists():
                st.video(seg.video_path)
            st.divider()


def render_video_result(result: VideoGenerationResult):
    """Render the generated video result."""
    st.markdown("## 🎥 Generated Video")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video player
        if Path(result.video_path).exists():
            st.video(result.video_path)
        else:
            st.warning(f"Video file not found: {result.video_path}")
    
    with col2:
        # Video info
        st.markdown("### Video Info")
        st.markdown(f"**Resolution:** {result.width}×{result.height}")
        st.markdown(f"**Duration:** {result.duration_seconds:.2f}s")
        st.markdown(f"**Frame Rate:** {result.fps} fps")
        st.markdown(f"**Prompt ID:** `{result.prompt_id[:8]}...`")
        
        # Download button
        if Path(result.video_path).exists():
            with open(result.video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Video",
                    data=f,
                    file_name=Path(result.video_path).name,
                    mime="video/mp4",
                    use_container_width=True
                )
        
        # Audio download if available
        if result.audio_path and Path(result.audio_path).exists():
            with open(result.audio_path, "rb") as f:
                st.download_button(
                    label="🔊 Download Audio",
                    data=f,
                    file_name=Path(result.audio_path).name,
                    mime="audio/wav",
                    use_container_width=True
                )


def render_history():
    """Render generation history."""
    if st.session_state.generated_videos:
        with st.expander("📜 Generation History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.generated_videos)):
                st.markdown(f"**{i+1}.** {item['prompt'][:50]}...")
                st.caption(f"Generated at {item['timestamp']}")
                if st.button(f"Load #{i+1}", key=f"load_{i}"):
                    st.session_state.last_result = item["result"]
                    st.rerun()
                st.divider()


def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar and get config values
    config_values = render_sidebar()
    
    # Render main content
    render_main_content(config_values)
    
    # Render history
    render_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "Powered by LTX-2 & ComfyUI | "
        "<a href='https://docs.ltx.video'>LTX Documentation</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
