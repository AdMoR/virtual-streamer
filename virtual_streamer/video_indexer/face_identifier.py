"""
Face identification implementation using face_recognition library.

Detects and recognizes faces in video frames by comparing against
known face encodings.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from virtual_streamer.video_indexer.interfaces import FaceIdentifier


class FaceRecognitionIdentifier(FaceIdentifier):
    """Face identifier using the face_recognition library.
    
    Uses dlib's face recognition model to detect faces and compare
    them against known face encodings.
    
    Attributes:
        tolerance: Distance threshold for face matching (lower = stricter).
        resolution: Resolution for frame resizing during processing.
    """

    def __init__(
        self,
        tolerance: float = 0.6,
        resolution: int = 480,
    ):
        """Initialize face identifier.
        
        Args:
            tolerance: Face matching tolerance (default: 0.6).
                      Lower values are more strict.
            resolution: Target resolution for processing (default: 480).
        """
        self.tolerance = tolerance
        self.resolution = resolution
        
        self._known_encodings: List[np.ndarray] = []
        self._known_names: List[str] = []
        self._name_to_encodings: Dict[str, List[np.ndarray]] = {}

    def load_known_faces(self, face_images: Dict[str, List[str]]) -> None:
        """Load known face encodings from image files.
        
        Args:
            face_images: Dictionary mapping person names to lists of image paths.
                        Example: {"fred": ["fred_1.jpg", "fred_2.jpg"], 
                                  "jamy": ["jamy_1.jpg"]}
        """
        self._known_encodings = []
        self._known_names = []
        self._name_to_encodings = {}
        
        for name, image_paths in face_images.items():
            self._name_to_encodings[name] = []
            
            for image_path in image_paths:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    encoding = encodings[0]
                    self._known_encodings.append(encoding)
                    self._known_names.append(name)
                    self._name_to_encodings[name].append(encoding)

    def load_known_faces_from_encodings(
        self, face_encodings: Dict[str, List[np.ndarray]]
    ) -> None:
        """Load known face encodings directly.
        
        Args:
            face_encodings: Dictionary mapping person names to lists of encodings.
        """
        self._known_encodings = []
        self._known_names = []
        self._name_to_encodings = face_encodings.copy()
        
        for name, encodings in face_encodings.items():
            for encoding in encodings:
                self._known_encodings.append(encoding)
                self._known_names.append(name)

    def identify(
        self, video_path: str, speed_up_factor: int = 4
    ) -> List[Tuple[str, int]]:
        """Identify faces in a video.
        
        Args:
            video_path: Path to video file.
            speed_up_factor: Process every Nth second of video (default: 4).
            
        Returns:
            List of (person_name, frame_index) tuples for each identified face.
        """
        if not self._known_encodings:
            return []
        
        video_stream = cv2.VideoCapture(video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 30.0  # Default fallback
        
        results: List[Tuple[str, int]] = []
        index = 0
        
        while True:
            index += 1
            still_reading, frame = video_stream.read()
            
            if not still_reading:
                break
            
            # Sample at intervals
            if index % (fps * speed_up_factor) != 1:
                continue
            
            # Find faces in this frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            for face_encoding in face_encodings:
                # Compare against known faces
                matches = face_recognition.compare_faces(
                    self._known_encodings, face_encoding, tolerance=self.tolerance
                )
                
                if not any(matches):
                    continue
                
                # Find best match
                face_distances = face_recognition.face_distance(
                    self._known_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self._known_names[best_match_index]
                    results.append((name, index))
        
        video_stream.release()
        return results

    def identify_frame(self, frame: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Identify faces in a single frame.
        
        Args:
            frame: Video frame as numpy array (H, W, C).
            
        Returns:
            List of (person_name, bounding_box) tuples.
            Bounding box is (top, right, bottom, left).
        """
        if not self._known_encodings:
            return []
        
        results: List[Tuple[str, Tuple[int, int, int, int]]] = []
        
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_location, face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                self._known_encodings, face_encoding, tolerance=self.tolerance
            )
            
            name = "Unknown"
            
            if any(matches):
                face_distances = face_recognition.face_distance(
                    self._known_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self._known_names[best_match_index]
            
            results.append((name, face_location))
        
        return results

    @property
    def known_face_names(self) -> List[str]:
        """Return list of unique known face names."""
        return list(self._name_to_encodings.keys())

    def get_face_encoding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face encoding from an image file.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Face encoding as numpy array, or None if no face found.
        """
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            return encodings[0]
        return None

