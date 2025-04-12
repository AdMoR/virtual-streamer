from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
import httpx
import torch
import uuid
import os
import numpy as np
import cv2
import json
import time
import datetime
import subprocess
import shutil
from tqdm import tqdm
from virtual_streamer.workflows.character_setup import CHARACTERS
from virtual_streamer.wav2lip import audio
from virtual_streamer.wav2lip.main_logic import preprocess, Config, datagen, do_load, FaceDetectionGroup
from virtual_streamer.utils.utils import sanitize_str, txt_to_speech_call, combine_video_and_audio, add_subtitle, s3_upload, SubtitleMode
from virtual_streamer.workflows.prompts import PROMPT, PROMPT_FR, PROMPT_FR_3, PROMPT_FR_2, SARCASTIC_PROMPT_FR, \
    STAND_UP_PROMPT, SARCASTIC_STANDUP, VERY_SARCASTIC_STANDUP_PROMPT, VERY_SARCASTIC_PROMPT
# Import relevant models from video_server
from virtual_streamer.video_server.models import DialogueEntry


# --- Pydantic Models ---

# Models for the existing /process endpoint
class QuestionData(BaseModel):
    question: str = ""
    character_name: str = ""
    subtitle_mode: str = "NONE"
    name: str = "User"

class ProcessRequest(BaseModel):
    question: QuestionData
    gpt_response: str

class ProcessResponse(BaseModel):
    video_path: str
    s3_path: Optional[str] = None
    response_text: str

class HealthResponse(BaseModel):
    status: str
    device: str

class Wav2LipRequest(BaseModel):
    audio_path: str # Path accessible by the server
    character_name: str
    output_dir: Optional[str] = None # Optional: Specify where to save, otherwise use temp

class Wav2LipResponse(BaseModel):
    raw_video_path: str # Path to the generated video (no audio)

# Model for the /generate-tts endpoint response
class TTSApiResponse(BaseModel):
    entry_id: str
    audio_path: str # Path accessible by subsequent services (e.g., Wav2Lip)
    # Potentially add duration or other metadata if needed


# --- FastAPI App ---
app = FastAPI()

# Initialize global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
UPLOAD_BUCKET = os.environ.get("S3_BUCKET_URL", "default-bucket")
args = Config()
args.checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoints/Wav2Lip.pth")
print(f"Using checkpoint: {args.checkpoint_path}")
model, detector, detector_model = do_load(args.checkpoint_path, device)
mel_step_size = 16
temp_dir = "./temp"

# Initialize face detection groups
print('Reading video frames...')
face_detection_groups: Dict[str, FaceDetectionGroup] = dict()
for k, v in CHARACTERS.items():
    preprocess(args, v.video_clip_path, k, detector, face_detection_groups)


def wav2lip_exec(dirname: str, audio_path: str, det_results: FaceDetectionGroup):
    fps = 24

    full_frames = det_results.full_frames
    face_det_results = det_results.face_det_results

    tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(os.path.basename(audio_path))
    out_path = f'{dirname}/result.avi'
    batch_size = args.wav2lip_batch_size

    if not audio_path.endswith('.wav'):
        print('Extracting raw audio...')
        subprocess.check_call([
            "ffmpeg", "-y",
            "-i", audio_path,
            f"{dirname}/temp.wav",
        ])
        audio_path = 'temp/temp.wav'

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print(f"Length of mel chunks: {len(mel_chunks)}, length of frames {len(full_frames)}")

    full_frames = full_frames[:len(mel_chunks)]
    face_det_results = face_det_results[:len(mel_chunks)]
    gen = datagen(args, full_frames.copy(), mel_chunks, face_det_results.copy())

    for i, rez in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        print(i)
        (face_img_batch, mel_batch, frames, coords) = rez

        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(f'{dirname}/result.avi',
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        print("-")
        face_img_batch = torch.FloatTensor(np.transpose(face_img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        print("--")
        with torch.no_grad():
            #with torch.amp.autocast("cuda"):
            pred = model(mel_batch, face_img_batch)
        print("---")
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    print("wav2lip_exec Done")
    return out_path


async def process_video(question_data: QuestionData, gpt_response: str, base_url: str) -> Dict[str, Any]:
    dirname = os.environ.get("OUT_VIDEO_FOLDER", "./out_video_folder")
    os.makedirs(dirname, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)

    # Extract data from question model
    question_text = question_data.question
    character_name = question_data.character_name
    subtitle_mode = question_data.subtitle_mode
    name = question_data.name

    # --- Step 1: Get the audio for the response ---
    # This endpoint uses a fixed speaker for now. If it needs dynamic characters,
    # it would need modification or potentially call the new /generate-tts endpoint.
    # For now, keep the original direct TTS call logic for this specific workflow.
    audio_filename = f"response_{hash(gpt_response) % 100000}_{uuid.uuid4()}.wav"
    audio_outpath = os.path.join(temp_dir, audio_filename)
    try:
        # Using a fixed speaker for this endpoint's purpose
        fixed_speaker_id = "male-pt-3%0A" # Or fetch from config if needed
        print(f"Generating TTS for /process request with speaker: {fixed_speaker_id}")
        # Assuming txt_to_speech_call is synchronous for now
        txt_to_speech_call(gpt_response, fixed_speaker_id, audio_outpath)
        #audio_outpath = "/home/amor/Downloads/1_PèreFouras_true.wav" # Example override for testing
        if not os.path.exists(audio_outpath):
             raise HTTPException(status_code=500, detail="TTS call failed to produce audio file.")
    except Exception as e:
        print(f"Error during TTS call in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech generation failed: {e}")

    # --- Step 2: Call Wav2Lip endpoint ---
    wav2lip_request_payload = Wav2LipRequest(
        audio_path=os.path.abspath(audio_outpath), # Send absolute path
        character_name=character_name,
        # Let the wav2lip endpoint manage its own temp output location initially
        output_dir=None
    )
    wav2lip_url = f"{base_url.rstrip('/')}/wav2lip"
    raw_video_path = None
    s = time.time()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client: # Use a timeout matching gunicorn
            response = await client.post(wav2lip_url, json=wav2lip_request_payload.dict())
            response.raise_for_status() # Raise exception for 4xx/5xx responses
            wav2lip_response_data = response.json()
            raw_video_path = wav2lip_response_data.get("raw_video_path")
            if not raw_video_path or not os.path.exists(raw_video_path):
                raise HTTPException(status_code=500, detail="Wav2Lip endpoint did not return a valid video path.")
        print("Wav2Lip API call time:", time.time() - s)
    except httpx.RequestError as e:
        print(f"Error calling Wav2Lip endpoint {wav2lip_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Wav2Lip service request failed: {e}")
    except httpx.HTTPStatusError as e:
        print(f"Wav2Lip endpoint returned error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Wav2Lip service error: {e.response.text}")
    except Exception as e: # Catch other potential errors like JSON parsing
        print(f"Unexpected error processing Wav2Lip response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process Wav2Lip result: {e}")


    # --- Step 3: Recombination and add subtitles ---
    tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(question_text[:30])
    outfile_combined_path = os.path.join(temp_dir, f'result_combined_{tag}.mp4')
    try:
        # Assuming combine_video_and_audio is synchronous
        combine_video_and_audio(raw_video_path, audio_outpath, outfile_combined_path)
    except Exception as e:
        print(f"Error combining video and audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to combine video/audio: {e}")

    # Add subtitles if needed
    outfile_titled_path = outfile_combined_path # Default to combined path
    try:
        if subtitle_mode == "QUESTION":
            subtitle = f"Question de {name} : {question_text}"
            outfile_titled_path = os.path.join(temp_dir, f'result_titled_{tag}.mp4')
            # Assuming add_subtitle is synchronous
            add_subtitle(subtitle, outfile_combined_path, outfile_titled_path)
        elif subtitle_mode == "VOICE_SUBTITLE":
            subtitle = gpt_response
            outfile_titled_path = os.path.join(temp_dir, f'result_titled_{tag}.mp4')
            # Assuming add_subtitle is synchronous
            add_subtitle(subtitle, outfile_combined_path, outfile_titled_path)
    except Exception as e:
        print(f"Error adding subtitles: {e}")
        # Non-fatal? Continue with the combined video if subtitling fails?
        # For now, let's raise an error.
        raise HTTPException(status_code=500, detail=f"Failed to add subtitles: {e}")

    # --- Step 4: Move the file to final location ---
    final_outfile_path = os.path.join(dirname, f"result_{tag}.mp4")
    try:
        shutil.move(outfile_titled_path, final_outfile_path) # Use move instead of copy
    except Exception as e:
        print(f"Error moving final video file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save final video: {e}")


    # --- Step 5: Upload to S3 if needed ---
    s3_path = None
    if UPLOAD_BUCKET != "default-bucket":
        try:
            # Assuming s3_upload is synchronous
            s3_path = s3_upload(final_outfile_path, UPLOAD_BUCKET)
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            # Decide if upload failure is critical. Maybe just log and continue?
            # For now, let's make it non-critical.
            pass # Logged error, but don't fail the request


    # --- Step 6: Cleanup temporary files ---
    # Clean up intermediate files from temp_dir (raw video, combined video, audio)
    # Be careful not to delete files needed elsewhere if temp_dir is shared.
    try:
        if os.path.exists(audio_outpath): os.remove(audio_outpath)
        # The raw_video_path might be inside a temp dir created by /wav2lip endpoint.
        # If that endpoint cleans up its own temp dir, we don't need to delete raw_video_path.
        # If it doesn't, we might need to delete it here, but need to know its location.
        # Assuming /wav2lip cleans up its own temp files if output_dir wasn't specified.
        # If raw_video_path was placed directly in *our* temp_dir (e.g., if output_dir was set), delete it.
        if raw_video_path and os.path.dirname(raw_video_path) == os.path.abspath(temp_dir) and os.path.exists(raw_video_path):
             os.remove(raw_video_path)
        if outfile_combined_path != outfile_titled_path and os.path.exists(outfile_combined_path):
             os.remove(outfile_combined_path)
        # outfile_titled_path was moved, not copied, so no need to delete original.
    except OSError as e:
        print(f"Warning: Error during temporary file cleanup: {e}")


    # --- Step 7: Return response ---
    return ProcessResponse(
        video_path=final_outfile_path, # Path on the server's filesystem
        s3_path=s3_path,
        response_text=gpt_response
    )


@app.post("/wav2lip", response_model=Wav2LipResponse)
async def run_wav2lip(payload: Wav2LipRequest):
    """
    Runs Wav2Lip generation on a given audio file and character.
    Returns the path to the raw generated video (without audio combined).
    """
    print(f"Received Wav2Lip request: {payload}")
    character_name = payload.character_name
    audio_path = payload.audio_path
    output_dir = payload.output_dir

    # Basic validation
    if not os.path.exists(audio_path):
         raise HTTPException(status_code=400, detail=f"Audio file not found at path: {audio_path}")

    # Determine output directory
    if output_dir:
        run_dirname = output_dir
        os.makedirs(run_dirname, exist_ok=True)
    else:
        # Create a unique temporary directory for this run
        run_dirname = f"./temp/wav2lip_run_{uuid.uuid4()}"
        os.makedirs(run_dirname, exist_ok=True)

    # Get the face detection group for the character
    if character_name not in face_detection_groups:
        # Handle case where character is not precomputed
        if character_name in CHARACTERS:
            character = CHARACTERS[character_name]
            # Note: Preprocessing here might be slow, ideally done beforehand
            print(f"Warning: Preprocessing character '{character_name}' on the fly.")
            face_det_group = preprocess(args, character.video_clip_path, character_name, detector, None)
            if face_det_group is None: # Check if preprocess failed
                 shutil.rmtree(run_dirname, ignore_errors=True) # Clean up temp dir
                 raise HTTPException(status_code=500, detail=f"Failed to preprocess character: {character_name}")
        else:
            # Character not found
            shutil.rmtree(run_dirname, ignore_errors=True) # Clean up temp dir
            raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found or preprocessed.")
    else:
        face_det_group = face_detection_groups[character_name]

    # Wav2lip video generation
    s = time.time()
    try:
        # Use a generic question string as it's only used for tagging output files inside wav2lip_exec
        # The actual output path is determined here.
        print("wav2lip started")
        raw_outfile_path = wav2lip_exec(run_dirname, audio_path, face_det_group)
        print("wav2lip prediction time:", time.time() - s)
    except Exception as e:
        shutil.rmtree(run_dirname, ignore_errors=True) # Clean up temp dir on error
        print(f"Error during wav2lip_exec: {e}")
        raise HTTPException(status_code=500, detail=f"Wav2Lip execution failed: {e}")

    # If output_dir was not specified, the result is in the temp dir.
    # The caller might want to move/delete it. For now, just return the path.
    # If output_dir *was* specified, the result is already there.

    return Wav2LipResponse(raw_video_path=raw_outfile_path)


@app.post("/generate-tts", response_model=TTSApiResponse)
async def generate_tts(payload: DialogueEntry):
    """
    Generates Text-to-Speech audio for a given dialogue entry.
    Expects a DialogueEntry object as the request body.
    """
    print(f"Received TTS generation request for entry: {payload.entry_id}")

    # Validate character_id and get speaker info
    if payload.character_id not in CHARACTERS:
        raise HTTPException(status_code=404, detail=f"Character ID '{payload.character_id}' not found.")

    # --- Assumption: CHARACTERS[character_id] has a 'speaker_id' attribute ---
    # Replace 'speaker_id' with the actual attribute name if different.
    # If no such mapping exists, this logic needs adjustment.
    try:
        # TODO: Confirm the actual attribute name for the speaker identifier in CHARACTERS
        speaker_id = CHARACTERS[payload.character_id].speaker_id
    except AttributeError:
         print(f"Error: Character '{payload.character_id}' found but missing 'speaker_id' attribute.")
         raise HTTPException(status_code=500, detail=f"Configuration error: Speaker ID not found for character '{payload.character_id}'.")
    except Exception as e:
         # Catch other potential errors accessing CHARACTERS
         print(f"Error accessing speaker info for character '{payload.character_id}': {e}")
         raise HTTPException(status_code=500, detail=f"Internal error retrieving character speaker info.")


    # Generate a unique filename in the temp directory
    # Using entry_id and a UUID ensures uniqueness and traceability
    audio_filename = f"tts_{payload.entry_id}_{uuid.uuid4()}.wav"
    audio_outpath = os.path.join(temp_dir, audio_filename)
    os.makedirs(temp_dir, exist_ok=True) # Ensure temp dir exists

    try:
        print(f"Generating TTS for entry {payload.entry_id} with speaker {speaker_id}...")
        # Assuming txt_to_speech_call is synchronous
        txt_to_speech_call(payload.text, speaker_id, audio_outpath)

        if not os.path.exists(audio_outpath):
             raise HTTPException(status_code=500, detail="TTS call failed to produce audio file.")
        print(f"TTS audio generated successfully at: {audio_outpath}")

    except Exception as e:
        print(f"Error during TTS call for entry {payload.entry_id}: {e}")
        # Clean up potentially empty file if created
        if os.path.exists(audio_outpath):
            try:
                os.remove(audio_outpath)
            except OSError:
                pass # Ignore cleanup error
        raise HTTPException(status_code=500, detail=f"Text-to-speech generation failed: {e}")

    # Return the path relative to the server or an absolute path
    # depending on how the next service (e.g., Wav2Lip) accesses files.
    # Using absolute path for clarity here.
    return TTSApiResponse(entry_id=payload.entry_id, audio_path=os.path.abspath(audio_outpath))


@app.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest, request: Request):
    """
    Process the request to generate a video response.
    """
    print(f"Received request: {payload}")

    # Extract data from request model
    question_data = payload.question
    gpt_response = payload.gpt_response

    # Process the video (run potentially long-running task in background if needed)
    # For now, running synchronously as the original code did
    base_url = str(request.base_url) # Get base URL (e.g., "http://localhost:5000/")
    result = await process_video(question_data, gpt_response, base_url)

    return result


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthResponse(status="healthy", device=device)
