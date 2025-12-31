"""
Florence-2 video description implementation.

Uses Microsoft's Florence-2 model to generate text descriptions
of video content by sampling and captioning frames.
"""

from typing import List, Optional

import cv2
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from virtual_streamer.video_indexer.interfaces import VideoDescriber


class FlorenceDescriber(VideoDescriber):
    """Video describer using Microsoft's Florence-2 model.
    
    Generates text descriptions of video content by sampling frames
    and using Florence-2's captioning capabilities.
    
    Attributes:
        model_id: HuggingFace model ID for Florence-2.
        task_prompt: Task prompt for Florence-2 (e.g., "<MORE_DETAILED_CAPTION>").
        resolution: Target resolution for frame resizing.
        device: Device to run the model on.
    """

    # Available task prompts for Florence-2
    TASK_CAPTION = "<CAPTION>"
    TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
    TASK_MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    TASK_OCR = "<OCR>"
    TASK_DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        task_prompt: str = "<MORE_DETAILED_CAPTION>",
        resolution: int = 480,
        device: Optional[str] = None,
    ):
        """Initialize Florence describer.
        
        Args:
            model_id: HuggingFace model ID (default: microsoft/Florence-2-large).
            task_prompt: Florence-2 task prompt for captioning.
            resolution: Target resolution for frame resizing (default: 480).
            device: Device to use (default: auto-detect cuda/cpu).
        """
        self.model_id = model_id
        self.task_prompt = task_prompt
        self.resolution = resolution
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Lazy loading - model loaded on first use
        self._model: Optional[AutoModelForCausalLM] = None
        self._processor: Optional[AutoProcessor] = None

    def _load_model(self) -> None:
        """Load Florence-2 model and processor (lazy loading)."""
        if self._model is not None:
            return
        
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype="auto"
            )
            .eval()
            .to(self.device)
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def _run_inference(self, frame: np.ndarray, text_input: Optional[str] = None) -> str:
        """Run Florence-2 inference on a single frame.
        
        Args:
            frame: Video frame as numpy array (H, W, C) in BGR format.
            text_input: Optional text input to append to task prompt.
            
        Returns:
            Generated caption text.
        """
        self._load_model()
        
        # Prepare prompt
        prompt = self.task_prompt
        if text_input is not None:
            prompt = prompt + text_input
        
        # Florence expects RGB, OpenCV gives BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Prepare inputs
        inputs = self._processor(
            text=prompt, images=frame_rgb, return_tensors="pt"
        ).to(self.device, torch.float16)
        
        # Generate
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        
        # Decode
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        # Post-process
        parsed_answer = self._processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(frame.shape[1], frame.shape[0])
        )
        
        # Extract the caption from the parsed result
        if isinstance(parsed_answer, dict):
            # Get the first value from the dict (usually keyed by task prompt)
            caption = list(parsed_answer.values())[0]
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
        else:
            caption = str(parsed_answer)
        
        return caption

    def _sample_frames(
        self, video_path: str, num_samples: int, speed_up_factor: int = 5
    ) -> List[np.ndarray]:
        """Sample frames from a video.
        
        Args:
            video_path: Path to video file.
            num_samples: Maximum number of frames to sample.
            speed_up_factor: Sample every Nth second (default: 5).
            
        Returns:
            List of sampled frames as numpy arrays.
        """
        video_stream = cv2.VideoCapture(video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 30.0  # Default fallback
        
        frames = []
        index = 0
        
        while True:
            index += 1
            still_reading, frame = video_stream.read()
            
            if not still_reading:
                break
            
            # Sample at intervals, but always include first frame
            if index % (fps * speed_up_factor) != 0 and len(frames) > 0:
                continue
            
            # Resize frame
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(
                frame, (int(self.resolution * aspect_ratio), self.resolution)
            )
            frames.append(frame)
            
            if len(frames) >= num_samples:
                break
        
        video_stream.release()
        return frames

    def describe(self, video_path: str) -> str:
        """Generate a combined text description for a video.
        
        Samples multiple frames and combines their descriptions.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Combined text description of the video content.
        """
        descriptions = self.describe_frames(video_path, num_samples=5)
        
        if not descriptions:
            return ""
        
        # Combine descriptions, removing duplicates
        unique_descriptions = list(dict.fromkeys(descriptions))
        return " | ".join(unique_descriptions)

    def describe_frames(self, video_path: str, num_samples: int = 5) -> List[str]:
        """Generate descriptions for sampled frames from a video.
        
        Args:
            video_path: Path to video file.
            num_samples: Number of frames to sample and describe.
            
        Returns:
            List of descriptions, one per sampled frame.
        """
        frames = self._sample_frames(video_path, num_samples)
        
        descriptions = []
        for frame in frames:
            caption = self._run_inference(frame)
            descriptions.append(caption)
        
        return descriptions

    def describe_frame(self, frame: np.ndarray) -> str:
        """Generate description for a single frame.
        
        Args:
            frame: Video frame as numpy array (H, W, C).
            
        Returns:
            Text description of the frame.
        """
        return self._run_inference(frame)

