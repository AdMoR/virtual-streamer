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
    "Jesus": CharacterConfig("Jesus", f"{project_root}/reference_videos/reference.mp4", "fr_0", "fr"),
    "Jesus_fr": CharacterConfig("Jesus", f"{project_root}/reference_videos/reference.mp4", "fr_0", "fr"),
    "Jesus_de": CharacterConfig("Jesus", f"{project_root}/reference_videos/reference.mp4", "friedrich", "de"),
    "Jesus_en": CharacterConfig("Jesus", f"{project_root}/reference_videos/reference.mp4", "en_0", "en"),
    "German Man": CharacterConfig("German", f"{project_root}/reference_videos/oktoberfest_man.mp4", "friedrich", "de"),
    "English DE man": CharacterConfig("English", f"{project_root}/reference_videos/oktoberfest_man.mp4", "en_0", "en")
}


TTSSettings = {
    "fr": [f"fr_{i}" for i in range(6)],
    "es": [f"es_{i}" for i in range(3)],
    "en": [f"en_{i}" for i in range(118)],
    "de": ["eva_k", "friedrich", "hokuspokus", "karlsson"],
}
