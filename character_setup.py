import enum
import dataclasses
import os.path


@dataclasses.dataclass
class CharacterConfig:
    name: str
    video_clip_path: str
    voice: str
    language: str


project_root = os.path.dirname(__file__)


CHARACTERS = {
    "Jesus": CharacterConfig("Jesus", f"{project_root}/assets/reference.mp4", "fr_1", "fr"),
    "Jamie": CharacterConfig("Jamie", f"{project_root}/assets/jamie_moon_landing.mp4", "fr_1", "fr"),
    "de": CharacterConfig("de", f"{project_root}/assets/reference.mp4", "friedrich", "de"),
}


TTSSettings = {
    "fr": [f"fr_{i}" for i in range(6)],
    "es": [f"es_{i}" for i in range(3)],
    "en": [f"en_{i}" for i in range(118)],
    "de": ["eva_k", "friedrich", "hokuspokus", "karlsson"],
}
