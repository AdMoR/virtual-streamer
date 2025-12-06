import math, time
import cv2
import torch
import numpy as np
from models import Wav2Lip


def load_model(path, device):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def face_detect(detector, images, batch_size):
    results = []
    pady1, pady2, padx1, padx2 = [0, 10, 0, 0]

    s = time.time()

    for image, rect in zip(images, face_rect(detector, images, batch_size)):
        if rect is None:
            cv2.imwrite(
                "temp/faulty_frame.jpg", image
            )  # check this frame where the face was not detected.
            raise ValueError(
                "Face not detected! Ensure the video contains a face in all the frames."
            )

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    print("face detect time:", time.time() - s)

    boxes = np.array(results)
    # if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    return results


def face_rect(detector, images, face_batch_size):
    num_batches = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    for i in range(num_batches):
        batch = images[i * face_batch_size : (i + 1) * face_batch_size]
        all_faces = detector(batch)  # return faces list of all images
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]
                prev_ret = tuple(map(int, box))
            yield prev_ret


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def _load(checkpoint_path, device):
    if device == "cuda":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint
