import dataclasses
import enum
import os
from typing import Callable, Dict, Any, List
from collections import defaultdict
import numpy as np
import cv2
from batch_face import RetinaFace
from virtual_streamer.wav2lip.model_utils import face_detect, load_model


@dataclasses.dataclass()
class Config:
    checkpoint_path: str = "./checkpoints/Wav2Lip.pth"
    resolution: int = 720
    wav2lip_batch_size: int = 8
    face_batch_size: int = 64
    img_size: int = 96

    static: bool = False
    pads: List[int] = (0, 10, 0, 0)
    nosmooth: bool = False
    rotate: bool = False
    box: List[int] = (-1, -1, -1, -1)
    crop: List[int] = (0, -1, 0, -1)
    resize_factor: float = 1.
    fps: int = 25
    outfile: str = 'results/result_voice.mp4'


@dataclasses.dataclass
class FaceDetectionGroup:
    full_frames: List[Any]
    face_det_results: List[Any]



def datagen(args: Config, frames: List[Any], mels: List[np.array], face_det_results: List[Any]):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    def unload(img_batch, mel_batch):
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        return img_batch, mel_batch, frame_batch, coords_batch

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            yield unload(img_batch, mel_batch)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        yield unload(img_batch, mel_batch)


def preprocess(args: Config, video_path: str, name: str, detector: Callable[Any, Any],
               face_detection_groups: Dict[str, FaceDetectionGroup]):
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = list()

    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break

        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(args.resolution * aspect_ratio), args.resolution))
        # if args.resize_factor > 1:
        #     frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

        y1, y2, x1, x2 = [0, -1, 0, -1]
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]
        full_frames.append(frame)

    face_det_results_origin = face_detect(detector, full_frames, args.face_batch_size)
    face_detection_groups[name] = FaceDetectionGroup(full_frames, face_det_results_origin)


def do_load(checkpoint_path, device):
    model = load_model(checkpoint_path, device)
    # SFDDetector.load_model(device)
    detector = RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")
    detector_model = detector.model
    print("Models loaded")
    return model, detector, detector_model
