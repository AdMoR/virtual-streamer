# Virtual Streamer System Documentation

## Overview

This codebase implements a virtual streamer system that can process video and audio inputs, generate responses using AI models, and create lip-synced videos of virtual characters responding to user questions. The system integrates multiple components including face detection, speech-to-text, text-to-speech, language models, and video processing.

## System Architecture

The system is composed of several key components:

1. **Face Detection and Processing** - Detects and processes faces in videos
2. **Speech Processing** - Converts speech to text and text to speech
3. **Language Models** - Generates responses to user questions
4. **Video Generation** - Creates lip-synced videos of virtual characters
5. **Web Services** - Provides API endpoints for the system
6. **Workflow Management** - Orchestrates the processing pipeline

## Key Components

### Face Detection

Located in the `face_detection/` directory, this component handles face detection and alignment:

- `api.py` - Main API for face detection that provides the `FaceAlignment` class
- `detection/` - Contains detection algorithms including SFD face detector
- `models.py` - Neural network models for face processing including `FAN` (Face Alignment Network)

The face detection system uses a modular approach where different detectors can be plugged in, with SFD (S³FD: Single Shot Scale-invariant Face Detector) being the default implementation.

### Video Understanding

The `video_understanding_run.py` script handles video analysis with these capabilities:
- Frame sampling from videos using PyAV
- Face recognition and identification
- Video content analysis using LLaVA-OneVision model
- Audio transcription using Whisper
- Personality embedding for character recognition

### Inference Pipeline

The `inference.py` file manages the inference workflow:
- Receives questions from a message queue
- Generates responses using GPT models
- Calls the web service to process videos
- Uploads results to S3
- Returns responses to the requester

### Content Creation Interface

The `creation_interface.py` provides a Streamlit-based interface for:
- Text generation using LLMs
- Video search and selection
- Audio generation with voice cloning
- Video composition with synchronized audio and subtitles

## Processing Pipeline

1. **Input Processing**:
   - User submits a question via API or message queue
   - The question is parsed and formatted for the language model

2. **Response Generation**:
   - Language model (GPT) generates a response based on the question and character personality
   - The response is processed and formatted for video generation

3. **Video Generation**:
   - Text is split into manageable segments
   - For each segment:
     - Appropriate video clip is selected
     - Text-to-speech converts the segment to audio
     - Face detection identifies faces in the video
     - Lip synchronization aligns the audio with the video
     - Subtitles are added

4. **Output Delivery**:
   - Final video is assembled from segments
   - Video is uploaded to S3 or served via web API
   - Response is sent back to the requester

## Models Used

1. **Face Detection Models**:
   - FAN (Face Alignment Network) for facial landmark detection
   - SFD (Single Shot Scale-invariant Face Detector) for face detection

2. **Video Understanding Models**:
   - LLaVA-OneVision for video content analysis
   - Whisper for audio transcription
   - Face recognition for character identification

3. **Language Models**:
   - GPT models for response generation
   - Claude models (Anthropic) for content creation

4. **Speech Models**:
   - Text-to-speech with voice cloning capabilities

## Implementation Details

### Face Detection

The face detection system uses a hierarchical approach:
1. Detect faces in frames using SFD
2. Process detected faces for further analysis
3. Return bounding boxes for detected faces

```python
# Example from face_detection/api.py
def get_detections_for_batch(self, images):
    images = images[..., ::-1]
    detected_faces = self.face_detector.detect_from_batch(images.copy())
    results = []

    for i, d in enumerate(detected_faces):
        if len(d) == 0:
            results.append(None)
            continue
        d = d[0]
        d = np.clip(d, 0, None)
        
        x1, y1, x2, y2 = map(int, d[:-1])
        results.append((x1, y1, x2, y2))

    return results
```

### Video Understanding

The video understanding system:
1. Samples frames from videos
2. Processes frames with AI models
3. Generates descriptions and identifies people

```python
# Example from video_understanding_run.py
def video_process_inference(my_video_path):
    container = av.open(my_video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)
    
    # For videos we have to feed a "video" type instead of "image"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Explain the action and background of this video in a very detailled manner."},
                ],
        },
    ]
    prompt = video_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = video_processor(videos=list(video), text=prompt, return_tensors="pt").to("cuda:0", torch.float16)
    out = video_model.generate(**inputs, max_new_tokens=100)
    rez = video_processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return rez[0].split("assistant")[1].strip()
```

### Content Creation

The content creation interface provides a workflow for:
1. Generating script text using LLMs
2. Searching for appropriate video clips
3. Generating audio with voice cloning
4. Combining everything into a final video

```python
# Example from creation_interface.py
def tab4_ui():
    st.title("Combined Results")
    if "sentences" not in st.session_state:
        compute_generated_sentences()
    generated_sentences = st.session_state["sentences"]
    video_chunks = list()

    ready = st.toggle("create final video", value=False)

    if ready:
        for i, sentence in enumerate(generated_sentences):
            selected_audio_id = build_id("selected_audio", sentence, i)
            selected_video_id = build_id("selected_video", sentence, i)
            if selected_audio_id in st.session_state and selected_video_id in st.session_state:
                video = st.session_state[selected_video_id]
                audio = st.session_state[selected_audio_id]
                outfile = f"./temp_{build_id('gen', sentence, i)}.mp4"
                combine_video_and_short_audio(video, audio, outfile)
                outfile_bis = f"./temp_{build_id('gen_sub', sentence, i)}.mp4"
                srt_path = f"./{i}.srt"
                result = model.transcribe(audio)
                result.to_srt_vtt(srt_path)
                add_subtitle_from_srt(outfile, srt_path, outfile_bis)
                video_chunks.append(outfile_bis)
```

## Deployment and Scaling

The system is designed to be deployed in a distributed environment:
- Web services for processing requests
- Message queues for asynchronous processing
- S3 for storage of videos and results
- GPU acceleration for model inference

## Future Improvements

Potential areas for enhancement:
1. Improved face detection accuracy
2. Better lip synchronization
3. More diverse character personalities
4. Real-time streaming capabilities
5. Multi-language support
6. Enhanced video quality and effects
