import enum
import dataclasses

@dataclasses.dataclass
class CharacterConfig:
    name: str
    video_clip_path: str
    voice: str
    language: str


CHARACTERS = {
    "Jesus": CharacterConfig("Jesus", "/media/amor/Storage/code_dw/cog-Wav2Lip/reference_videos/reference.mp4", "fr_0", "fr"),
    "Jesus_fr": CharacterConfig("Jesus", "/media/amor/Storage/code_dw/cog-Wav2Lip/reference_videos/reference.mp4", "fr_0", "fr"),
    "Jesus_de": CharacterConfig("Jesus", "/media/amor/Storage/code_dw/cog-Wav2Lip/reference_videos/reference.mp4", "friedrich", "de"),
    "Jesus_en": CharacterConfig("Jesus", "/media/amor/Storage/code_dw/cog-Wav2Lip/reference_videos/reference.mp4", "en_0", "en"),
    "German Man": CharacterConfig("German", "/media/amor/Storage/code_dw/cog-Wav2Lip/reference_videos/oktoberfest_man.mp4", "friedrich", "de"),
    "English DE man": CharacterConfig("English", "/media/amor/Storage/code_dw/cog-Wav2Lip/reference_videos/oktoberfest_man.mp4", "en_0", "en")
}


TTSSettings = {
    "fr": [f"fr_{i}" for i in range(6)],
    "es": [f"es_{i}" for i in range(3)],
    "en": [f"en_{i}" for i in range(118)],
    "de": ["eva_k", "friedrich", "hokuspokus", "karlsson"],
}
