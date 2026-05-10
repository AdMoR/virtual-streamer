"""
Shared pytest fixtures for the virtual-streamer test suite.

Mocked dependencies:
  - WanGPLTXClient       → mock_ltx_client  (patches class in story_to_video module)
  - StableDiffusionCppClient → mock_sd_client (patches class in story_to_video module)
  - get_storage_client   → mock_storage_client (patches both top-level and lazy usages)
  - get_entity_repository → mock_entity_repository (patches source module)
  - get_length           → mock_get_length (patches source module)
  - subprocess.run       → mock_subprocess_run (patches story_to_video.subprocess.run)
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_streamer.image_generation.models import Camera, FluxPrompt
from virtual_streamer.image_generation.stable_cpp_client import ImageGenerationResult
from virtual_streamer.video_generation.config import DialogLine, StoryOutput
from virtual_streamer.video_generation.ltx_client import (
    LTXVideoConfig,
    VideoGenerationParams,
    VideoGenerationResult,
)
from virtual_streamer.video_generation.scene_input import SceneInput, StoryInput


# ---------------------------------------------------------------------------
# Fake filesystem helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_video_file(tmp_path) -> str:
    path = tmp_path / "segment.mp4"
    path.write_bytes(b"FAKE_VIDEO")
    return str(path)


@pytest.fixture
def fake_audio_file(tmp_path) -> str:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"FAKE_AUDIO")
    return str(path)


@pytest.fixture
def fake_image_file(tmp_path) -> str:
    path = tmp_path / "image.png"
    path.write_bytes(b"FAKE_IMAGE")
    return str(path)


@pytest.fixture
def output_dir(tmp_path) -> str:
    return str(tmp_path / "output")


# ---------------------------------------------------------------------------
# Sample domain objects
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dialog_line() -> DialogLine:
    return DialogLine(
        character_id="fred",
        text="Eh dis donc Jamy!",
        scene_description=FluxPrompt(
            scene="science lab",
            subjects=[],
            lighting="soft",
            camera=Camera(angle="eye level", distance="medium shot"),
        ),
        location_id=None,
    )


@pytest.fixture
def sample_story_output(sample_dialog_line) -> StoryOutput:
    return StoryOutput(
        title="Test Story",
        story_plan="A test story plan.",
        dialog=[sample_dialog_line, sample_dialog_line],
    )


@pytest.fixture
def sample_video_params() -> VideoGenerationParams:
    return VideoGenerationParams(
        prompt="",
        duration_seconds=3.0,
        width=1280,
        height=720,
        fps=24,
        steps=4,
        cfg_scale=3.0,
    )


@pytest.fixture
def sample_ltx_config() -> LTXVideoConfig:
    return LTXVideoConfig(server_url="http://mock-server:8082")


@pytest.fixture
def sample_scene_input() -> SceneInput:
    return SceneInput(
        scene_index=0,
        ltx_prompt="A scientist in a lab explaining something",
        speaker_id="fred",
        spoken_line="Eh dis donc Jamy!",
        location_id="loc-1",
        character_ids_on_screen=["fred"],
        scene_visual_description={
            "scene": "science lab",
            "subjects": [],
            "lighting": "soft",
            "camera": {"angle": "eye level", "distance": "medium shot"},
        },
        raw_scene_data={},
    )


@pytest.fixture
def sample_story_input(sample_scene_input) -> StoryInput:
    scene_1 = sample_scene_input.model_copy(update={"scene_index": 1})
    return StoryInput(
        title="Test Story",
        story_plan="A test story plan.",
        story_template_id="tmpl-1",
        raw_agent_output={},
        scenes=[sample_scene_input, scene_1],
    )


# ---------------------------------------------------------------------------
# Mock client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ltx_client(fake_video_file):
    """
    Patches WanGPLTXClient in story_to_video module.
    Yields the inner AsyncMock instance so tests can inspect .generate_video calls.
    """
    result = VideoGenerationResult(
        video_path=fake_video_file,
        duration_seconds=3.0,
        width=1280,
        height=720,
        fps=24,
        prompt_id="mock-pid",
    )
    mock_instance = AsyncMock()
    mock_instance.generate_video = AsyncMock(return_value=result)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)

    mock_cls = MagicMock(return_value=mock_instance)
    with patch("virtual_streamer.video_generation.story_to_video.WanGPLTXClient", mock_cls):
        yield mock_instance


@pytest.fixture
def mock_sd_client(fake_image_file):
    """
    Patches StableDiffusionCppClient in story_to_video module.
    Yields the inner AsyncMock instance so tests can inspect .txt2image / .image_edit calls.
    """
    result = ImageGenerationResult(
        image_path=fake_image_file,
        width=1280,
        height=720,
        seed=-1,
        prompt_id="img-mock-pid",
    )
    mock_instance = AsyncMock()
    mock_instance.txt2image = AsyncMock(return_value=result)
    mock_instance.image_edit = AsyncMock(return_value=result)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)

    mock_cls = MagicMock(return_value=mock_instance)
    with patch("virtual_streamer.video_generation.story_to_video.StableDiffusionCppClient", mock_cls):
        yield mock_instance


@pytest.fixture
def mock_storage_client():
    """
    Patches get_storage_client at both the top-level import and the lazy import
    inside story_to_video function bodies.
    Yields the AsyncMock storage instance.
    """
    storage = AsyncMock()
    storage.upload_file = AsyncMock(return_value=None)
    storage.put_json = AsyncMock(return_value=None)

    async def _download(src, dst):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"DOWNLOADED")

    storage.download_file = AsyncMock(side_effect=_download)

    with patch("virtual_streamer.video_generation.story_to_video.get_storage_client", return_value=storage):
        with patch("virtual_streamer.utils.minio_client.get_storage_client", return_value=storage):
            yield storage


@pytest.fixture
def mock_entity_repository():
    """
    Patches get_entity_repository at the source module (lazy-imported inside function bodies).
    Yields the AsyncMock repo instance.
    """
    repo = AsyncMock()
    repo.get_location = AsyncMock(return_value={
        "location_id": "loc-1",
        "story_template_id": "tmpl-1",
        "description": "A test lab",
    })
    repo.get_character = AsyncMock(return_value={
        "character_id": "fred",
        "name": "Fred",
        "description": "a scientist",
        "identity_images": [],
    })
    repo.list_locations_by_template = AsyncMock(return_value=[{
        "location_id": "loc-1",
        "story_template_id": "tmpl-1",
        "description": "A test lab",
    }])

    with patch("virtual_streamer.utils.entity_repository.get_entity_repository", return_value=repo):
        yield repo


@pytest.fixture
def mock_get_length():
    """
    Patches get_length at the source module (lazy-imported inside generate_segment).
    Default return value is 3.5. Override with mock_get_length.return_value = x in tests.
    """
    with patch("virtual_streamer.utils.utils.get_length", return_value=3.5) as mock:
        yield mock


@pytest.fixture
def mock_subprocess_run(tmp_path):
    """
    Patches subprocess.run used by concatenate_videos.
    Side effect creates the output file (last arg of the ffmpeg command) so
    _file_size checks pass after the mock returns returncode=0.
    """
    def _create_output(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x00")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch(
        "virtual_streamer.video_generation.story_to_video.subprocess.run",
        side_effect=_create_output,
    ) as mock:
        yield mock
