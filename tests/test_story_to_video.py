"""
Tests for virtual_streamer/video_generation/story_to_video.py.

External dependencies mocked via conftest.py fixtures:
  - mock_ltx_client     → patches WanGPLTXClient class
  - mock_sd_client      → patches StableDiffusionCppClient class
  - mock_storage_client → patches get_storage_client (both import sites)
  - mock_entity_repository → patches get_entity_repository
  - mock_get_length     → patches get_length utility
  - mock_subprocess_run → patches subprocess.run (for ffmpeg calls)
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_streamer.video_generation.ltx_client import (
    LTXVideoConfig,
    VideoGenerationParams,
    VideoGenerationResult,
)
from virtual_streamer.video_generation.scene_input import SceneInput, StoryInput
from virtual_streamer.video_generation.story_to_video import (
    StoryVideoResult,
    _frames_from_duration,
    concatenate_videos,
    generate_segment_from_input,
    generate_scene_image_from_input,
    story_input_to_video,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_segment_client(fake_video_file: str) -> AsyncMock:
    """Return a bare AsyncMock LTX client for direct segment tests."""
    result = VideoGenerationResult(
        video_path=fake_video_file,
        duration_seconds=3.0,
        width=1280,
        height=720,
        fps=24,
        prompt_id="mock-pid",
    )
    client = AsyncMock()
    client.generate_video = AsyncMock(return_value=result)
    return client


# ---------------------------------------------------------------------------
# concatenate_videos
# ---------------------------------------------------------------------------

class TestConcatenateVideosSingleFile:

    def test_copies_directly_without_ffmpeg(self, fake_video_file, tmp_path, mock_subprocess_run):
        out = str(tmp_path / "out.mp4")
        result = concatenate_videos([fake_video_file], out, str(tmp_path))
        assert result == out
        assert Path(out).read_bytes() == b"FAKE_VIDEO"
        mock_subprocess_run.assert_not_called()

    def test_returns_output_path(self, fake_video_file, tmp_path):
        out = str(tmp_path / "out.mp4")
        result = concatenate_videos([fake_video_file], out, str(tmp_path))
        assert result == out


class TestConcatenateVideosMultipleFiles:

    def test_stream_copy_success(self, fake_video_file, tmp_path, mock_subprocess_run):
        out = str(tmp_path / "out.mp4")
        concatenate_videos([fake_video_file, fake_video_file], out, str(tmp_path))
        assert mock_subprocess_run.call_count == 1
        cmd = mock_subprocess_run.call_args[0][0]
        assert "-c" in cmd
        assert "copy" in cmd

    def test_fallback_reencode_on_stream_copy_failure(self, fake_video_file, tmp_path):
        out = str(tmp_path / "out.mp4")
        call_count = 0

        def _side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(returncode=1, stdout="", stderr="stream copy failed")
            Path(cmd[-1]).write_bytes(b"\x00")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("virtual_streamer.video_generation.story_to_video.subprocess.run", side_effect=_side_effect):
            concatenate_videos([fake_video_file, fake_video_file], out, str(tmp_path))

        assert call_count == 2

    def test_raises_runtime_error_if_both_fail(self, fake_video_file, tmp_path):
        out = str(tmp_path / "out.mp4")
        with patch(
            "virtual_streamer.video_generation.story_to_video.subprocess.run",
            return_value=MagicMock(returncode=1, stdout="", stderr="error"),
        ):
            with pytest.raises(RuntimeError):
                concatenate_videos([fake_video_file, fake_video_file], out, str(tmp_path))

    def test_raises_file_not_found_for_missing_segment(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            concatenate_videos(["/nonexistent/segment.mp4"], str(tmp_path / "out.mp4"), str(tmp_path))

    def test_raises_value_error_for_empty_segment(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(ValueError):
            concatenate_videos([str(empty)], str(tmp_path / "out.mp4"), str(tmp_path))

    def test_raises_runtime_error_for_empty_output(self, fake_video_file, tmp_path):
        out = str(tmp_path / "out.mp4")
        # subprocess returns success but does NOT create the output file
        with patch(
            "virtual_streamer.video_generation.story_to_video.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        ):
            with pytest.raises(RuntimeError, match="empty output"):
                concatenate_videos([fake_video_file, fake_video_file], out, str(tmp_path))

    def test_whitespace_in_path_is_sanitized(self, tmp_path, mock_subprocess_run):
        spaced = tmp_path / "my segment.mp4"
        spaced.write_bytes(b"FAKE_VIDEO")
        out = str(tmp_path / "out.mp4")
        concatenate_videos([str(spaced), str(spaced)], out, str(tmp_path))
        # The ffmpeg concat list should not contain spaces in filenames
        concat_text = None
        for call in mock_subprocess_run.call_args_list:
            cmd = call[0][0]
            for j, arg in enumerate(cmd):
                if arg == "-i" and j + 1 < len(cmd):
                    concat_path = cmd[j + 1]
                    if Path(concat_path).exists():
                        concat_text = Path(concat_path).read_text()
        if concat_text:
            for line in concat_text.splitlines():
                if line.startswith("file"):
                    path_part = line.split("'")[1] if "'" in line else line.split()[1]
                    assert " " not in path_part


# ---------------------------------------------------------------------------
# generate_segment_from_input
# ---------------------------------------------------------------------------

class TestGenerateSegmentFromInput:

    async def test_t2v_calls_generate_video_once(self, fake_video_file, sample_scene_input, sample_video_params, tmp_path):
        client = _make_segment_client(fake_video_file)
        await generate_segment_from_input(client, sample_scene_input, str(tmp_path), sample_video_params)
        client.generate_video.assert_called_once()

    async def test_t2v_no_image_no_audio_in_params(self, fake_video_file, sample_scene_input, sample_video_params, tmp_path):
        client = _make_segment_client(fake_video_file)
        await generate_segment_from_input(client, sample_scene_input, str(tmp_path), sample_video_params)
        params = client.generate_video.call_args.kwargs["params"]
        assert params.image_path is None
        assert params.audio_path is None
        assert params.enable_audio is False

    async def test_result_fields_populated(self, fake_video_file, sample_scene_input, sample_video_params, tmp_path):
        client = _make_segment_client(fake_video_file)
        scene = sample_scene_input.model_copy(update={"scene_index": 2})
        seg = await generate_segment_from_input(client, scene, str(tmp_path), sample_video_params)
        assert seg.index == 2
        assert seg.video_path == fake_video_file
        assert seg.prompt_id == "mock-pid"
        assert seg.duration_seconds == 3.0
        assert seg.audio_path is None
        assert seg.image_path is None

    async def test_segment_dir_created_with_uuid_suffix(self, fake_video_file, sample_scene_input, sample_video_params, tmp_path):
        client = _make_segment_client(fake_video_file)
        await generate_segment_from_input(client, sample_scene_input, str(tmp_path), sample_video_params)
        assert any(tmp_path.glob("scene_000_*"))

    async def test_i2v_image_path_forwarded(self, fake_video_file, fake_image_file, sample_scene_input, sample_video_params, tmp_path):
        client = _make_segment_client(fake_video_file)
        seg = await generate_segment_from_input(
            client, sample_scene_input, str(tmp_path), sample_video_params,
            image_path=fake_image_file,
        )
        params = client.generate_video.call_args.kwargs["params"]
        assert params.image_path == fake_image_file
        assert params.enable_audio is False
        assert seg.image_path == fake_image_file

    async def test_audio_conditioned_i2v_both_params_set(self, fake_video_file, fake_audio_file, fake_image_file, sample_scene_input, sample_video_params, tmp_path, mock_get_length):
        client = _make_segment_client(fake_video_file)
        seg = await generate_segment_from_input(
            client, sample_scene_input, str(tmp_path), sample_video_params,
            audio_path=fake_audio_file, image_path=fake_image_file,
        )
        params = client.generate_video.call_args.kwargs["params"]
        assert params.image_path == fake_image_file
        assert params.audio_path == fake_audio_file
        assert params.enable_audio is True
        assert seg.audio_path == fake_audio_file

    async def test_audio_adapts_duration_to_get_length(self, fake_video_file, fake_audio_file, sample_scene_input, sample_video_params, tmp_path, mock_get_length):
        mock_get_length.return_value = 5.5
        client = _make_segment_client(fake_video_file)
        await generate_segment_from_input(
            client, sample_scene_input, str(tmp_path), sample_video_params,
            audio_path=fake_audio_file,
        )
        params = client.generate_video.call_args.kwargs["params"]
        assert params.frames == _frames_from_duration(5.5 + 0.5, sample_video_params.fps)

    async def test_missing_audio_file_falls_back_to_configured_duration(self, fake_video_file, sample_scene_input, sample_video_params, tmp_path):
        client = _make_segment_client(fake_video_file)
        await generate_segment_from_input(
            client, sample_scene_input, str(tmp_path), sample_video_params,
            audio_path="/nonexistent/audio.wav",
        )
        params = client.generate_video.call_args.kwargs["params"]
        assert params.frames == _frames_from_duration(sample_video_params.duration_seconds, sample_video_params.fps)

    async def test_get_length_zero_falls_back_to_configured_duration(self, fake_video_file, fake_audio_file, sample_scene_input, sample_video_params, tmp_path, mock_get_length):
        mock_get_length.return_value = 0.0
        client = _make_segment_client(fake_video_file)
        await generate_segment_from_input(
            client, sample_scene_input, str(tmp_path), sample_video_params,
            audio_path=fake_audio_file,
        )
        params = client.generate_video.call_args.kwargs["params"]
        assert params.frames == _frames_from_duration(sample_video_params.duration_seconds, sample_video_params.fps)

    async def test_get_length_exception_falls_back_to_configured_duration(self, fake_video_file, fake_audio_file, sample_scene_input, sample_video_params, tmp_path, mock_get_length):
        mock_get_length.side_effect = RuntimeError("bad file")
        client = _make_segment_client(fake_video_file)
        await generate_segment_from_input(
            client, sample_scene_input, str(tmp_path), sample_video_params,
            audio_path=fake_audio_file,
        )
        params = client.generate_video.call_args.kwargs["params"]
        assert params.frames == _frames_from_duration(sample_video_params.duration_seconds, sample_video_params.fps)


# ---------------------------------------------------------------------------
# generate_scene_image_from_input
# ---------------------------------------------------------------------------

class TestGenerateSceneImageFromInput:

    async def test_no_refs_uses_txt2image(self, mock_sd_client, tmp_path, sample_video_params):
        scene = SceneInput(scene_index=0, ltx_prompt="A dark forest scene", raw_scene_data={})
        result = await generate_scene_image_from_input(
            scene_input=scene, location=None, character_dicts=[], output_dir=str(tmp_path),
            video_params=sample_video_params,
        )
        mock_sd_client.txt2image.assert_called_once()
        mock_sd_client.image_edit.assert_not_called()
        assert result is not None

    async def test_no_refs_people_excluded_from_negative_prompt(self, mock_sd_client, tmp_path, sample_video_params):
        scene = SceneInput(scene_index=0, ltx_prompt="Empty space station", raw_scene_data={})
        await generate_scene_image_from_input(
            scene_input=scene, location=None, character_dicts=[], output_dir=str(tmp_path),
            video_params=sample_video_params,
        )
        params = mock_sd_client.txt2image.call_args[0][0]
        assert "people" in params.negative_prompt

    async def test_location_with_image_path_uses_image_edit(self, mock_sd_client, mock_storage_client, tmp_path, sample_video_params):
        scene = SceneInput(scene_index=0, ltx_prompt="A lab scene", raw_scene_data={})
        location = {"location_id": "loc-1", "image_path": "minio/locations/lab.png"}
        result = await generate_scene_image_from_input(
            scene_input=scene, location=location, character_dicts=[], output_dir=str(tmp_path),
            video_params=sample_video_params,
        )
        mock_storage_client.download_file.assert_called_once()
        mock_sd_client.image_edit.assert_called_once()
        mock_sd_client.txt2image.assert_not_called()
        assert result is not None

    async def test_character_with_identity_images_uses_image_edit(self, mock_sd_client, mock_storage_client, tmp_path, sample_video_params):
        scene = SceneInput(scene_index=0, ltx_prompt="A lab scene", raw_scene_data={})
        char = {"character_id": "fred", "identity_images": ["minio/chars/fred.png"]}
        result = await generate_scene_image_from_input(
            scene_input=scene, location=None, character_dicts=[char], output_dir=str(tmp_path),
            video_params=sample_video_params,
        )
        mock_storage_client.download_file.assert_called_once()
        mock_sd_client.image_edit.assert_called_once()
        mock_sd_client.txt2image.assert_not_called()
        assert result is not None

    async def test_location_and_character_both_with_images_downloads_both(self, mock_sd_client, mock_storage_client, tmp_path, sample_video_params):
        scene = SceneInput(scene_index=0, ltx_prompt="A lab scene", raw_scene_data={})
        location = {"location_id": "loc-1", "image_path": "minio/locations/lab.png"}
        char = {"character_id": "fred", "identity_images": ["minio/chars/fred.png"]}
        await generate_scene_image_from_input(
            scene_input=scene, location=location, character_dicts=[char], output_dir=str(tmp_path),
            video_params=sample_video_params,
        )
        assert mock_storage_client.download_file.call_count == 2
        mock_sd_client.image_edit.assert_called_once()

    async def test_no_scene_visual_description_uses_ltx_prompt(self, mock_sd_client, tmp_path, sample_video_params):
        scene = SceneInput(
            scene_index=0,
            ltx_prompt="my specific prompt",
            scene_visual_description=None,
            raw_scene_data={},
        )
        await generate_scene_image_from_input(
            scene_input=scene, location=None, character_dicts=[], output_dir=str(tmp_path),
            video_params=sample_video_params,
        )
        params = mock_sd_client.txt2image.call_args[0][0]
        assert params.prompt == "my specific prompt"

    async def test_scene_visual_description_overrides_ltx_prompt(self, mock_sd_client, tmp_path, sample_video_params):
        scene = SceneInput(
            scene_index=0,
            ltx_prompt="this should NOT be used",
            scene_visual_description={
                "scene": "futuristic laboratory",
                "subjects": [],
                "lighting": "neon",
                "camera": {"angle": "eye level", "distance": "medium shot"},
            },
            raw_scene_data={},
        )
        await generate_scene_image_from_input(
            scene_input=scene, location=None, character_dicts=[], output_dir=str(tmp_path),
            video_params=sample_video_params,
        )
        params = mock_sd_client.txt2image.call_args[0][0]
        assert params.prompt != "this should NOT be used"
        assert "futuristic laboratory" in params.prompt

    async def test_download_failure_falls_back_to_txt2image(self, mock_sd_client, mock_storage_client, tmp_path, sample_video_params):
        mock_storage_client.download_file.side_effect = RuntimeError("MinIO unavailable")
        scene = SceneInput(scene_index=0, ltx_prompt="A lab scene", raw_scene_data={})
        location = {"location_id": "loc-1", "image_path": "minio/locations/lab.png"}
        await generate_scene_image_from_input(
            scene_input=scene, location=location, character_dicts=[], output_dir=str(tmp_path),
            video_params=sample_video_params,
        )
        mock_sd_client.txt2image.assert_called_once()
        mock_sd_client.image_edit.assert_not_called()

    async def test_sd_exception_returns_none(self, tmp_path, sample_video_params):
        with patch(
            "virtual_streamer.video_generation.story_to_video.StableDiffusionCppClient",
            side_effect=RuntimeError("SD server down"),
        ):
            scene = SceneInput(scene_index=0, ltx_prompt="A lab scene", raw_scene_data={})
            result = await generate_scene_image_from_input(
                scene_input=scene, location=None, character_dicts=[], output_dir=str(tmp_path),
                video_params=sample_video_params,
            )
        assert result is None


# ---------------------------------------------------------------------------
# story_input_to_video — basic
# ---------------------------------------------------------------------------

class TestStoryInputToVideoBasic:

    async def test_returns_story_video_result(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        result = await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        assert isinstance(result, StoryVideoResult)

    async def test_two_scenes_produce_two_segments(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        result = await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        assert len(result.segments) == 2

    async def test_story_title_preserved(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        result = await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        assert result.story_title == "Test Story"

    async def test_total_duration_positive(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        result = await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        assert result.total_duration_seconds > 0

    async def test_progress_callback_called_per_scene(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        progress = MagicMock()
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
            progress_callback=progress,
        )
        assert progress.call_count >= len(sample_story_input.scenes)


# ---------------------------------------------------------------------------
# story_input_to_video — conditioning (the primary ask)
# ---------------------------------------------------------------------------

class TestStoryInputToVideoConditioning:
    """
    Verify that the right generation mode (t2v / i2v / audio-conditioned i2v) is
    selected based on entity data, and that SD (SDCPP) is called correctly.
    """

    async def test_no_template_id_skips_entity_lookup(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        story = sample_story_input.model_copy(update={"story_template_id": None})
        await story_input_to_video(
            story_input=story,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        mock_entity_repository.list_locations_by_template.assert_not_called()
        mock_entity_repository.get_character.assert_not_called()
        # SD conditioning still runs (txt2image, no refs)
        mock_sd_client.txt2image.assert_called()
        # LTX receives image_path (from txt2image result → i2v, not t2v)
        for call in mock_ltx_client.generate_video.call_args_list:
            assert call.kwargs["params"].image_path is not None

    async def test_template_id_loads_entities_from_repo(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        mock_entity_repository.list_locations_by_template.assert_called_once_with("tmpl-1")
        assert mock_entity_repository.get_character.call_count >= 1

    async def test_location_with_image_path_triggers_image_edit_and_i2v(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_storage_client, mock_entity_repository):
        mock_entity_repository.list_locations_by_template.return_value = [{
            "location_id": "loc-1",
            "story_template_id": "tmpl-1",
            "description": "A test lab",
            "image_path": "minio/locations/loc-1.png",
        }]
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        mock_sd_client.image_edit.assert_called()
        mock_sd_client.txt2image.assert_not_called()
        for call in mock_ltx_client.generate_video.call_args_list:
            assert call.kwargs["params"].image_path is not None

    async def test_character_with_identity_images_triggers_image_edit_and_i2v(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_storage_client, mock_entity_repository):
        mock_entity_repository.get_character.return_value = {
            "character_id": "fred",
            "name": "Fred",
            "description": "a scientist",
            "identity_images": ["minio/chars/fred_001.png"],
        }
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        mock_sd_client.image_edit.assert_called()
        for call in mock_ltx_client.generate_video.call_args_list:
            assert call.kwargs["params"].image_path is not None

    async def test_location_and_character_both_with_images_downloads_all_refs(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_storage_client, mock_entity_repository):
        mock_entity_repository.list_locations_by_template.return_value = [{
            "location_id": "loc-1",
            "story_template_id": "tmpl-1",
            "description": "A test lab",
            "image_path": "minio/locations/loc-1.png",
        }]
        mock_entity_repository.get_character.return_value = {
            "character_id": "fred",
            "name": "Fred",
            "description": "a scientist",
            "identity_images": ["minio/chars/fred_001.png"],
        }
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        # 2 scenes × (1 location image + 1 character image) = 4 downloads
        assert mock_storage_client.download_file.call_count == 4
        mock_sd_client.image_edit.assert_called()

    async def test_entities_with_no_images_use_txt2image_but_still_i2v(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        # default mock_entity_repository: location has no image_path, character has no identity_images
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        mock_sd_client.txt2image.assert_called()
        mock_sd_client.image_edit.assert_not_called()
        # txt2image still produces an image → LTX is i2v, not t2v
        for call in mock_ltx_client.generate_video.call_args_list:
            assert call.kwargs["params"].image_path is not None

    async def test_sd_failure_falls_back_to_t2v(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        # Make both SD methods raise so generate_scene_image_from_input returns None
        mock_sd_client.txt2image.side_effect = RuntimeError("SD down")
        mock_sd_client.image_edit.side_effect = RuntimeError("SD down")
        result = await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        # Graceful degradation: generation still succeeds with t2v
        assert len(result.segments) == 2
        for call in mock_ltx_client.generate_video.call_args_list:
            assert call.kwargs["params"].image_path is None

    async def test_location_image_plus_audio_gives_audio_conditioned_i2v(self, fake_audio_file, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_storage_client, mock_entity_repository, mock_get_length):
        mock_entity_repository.list_locations_by_template.return_value = [{
            "location_id": "loc-1",
            "story_template_id": "tmpl-1",
            "description": "A test lab",
            "image_path": "minio/locations/loc-1.png",
        }]
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
            segment_audio_paths={0: fake_audio_file},
        )
        calls = mock_ltx_client.generate_video.call_args_list
        # Scene 0: audio-conditioned i2v
        first = calls[0].kwargs["params"]
        assert first.image_path is not None
        assert first.audio_path == fake_audio_file
        assert first.enable_audio is True
        # Scene 1: i2v without audio
        second = calls[1].kwargs["params"]
        assert second.image_path is not None
        assert second.audio_path is None


# ---------------------------------------------------------------------------
# story_input_to_video — resilience
# ---------------------------------------------------------------------------

class TestStoryInputToVideoResilience:

    async def test_failed_segment_is_skipped_rest_succeed(self, fake_video_file, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_subprocess_run, mock_sd_client, mock_entity_repository):
        ok_result = VideoGenerationResult(
            video_path=fake_video_file, duration_seconds=3.0,
            width=1280, height=720, fps=24, prompt_id="ok-pid",
        )
        mock_instance = AsyncMock()
        mock_instance.generate_video = AsyncMock(
            side_effect=[RuntimeError("GPU OOM"), ok_result]
        )
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("virtual_streamer.video_generation.story_to_video.WanGPLTXClient", MagicMock(return_value=mock_instance)):
            result = await story_input_to_video(
                story_input=sample_story_input,
                ltx_config=sample_ltx_config,
                video_params=sample_video_params,
                output_dir=output_dir,
            )
        assert len(result.segments) == 1

    async def test_all_segments_fail_raises_runtime_error(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_sd_client, mock_entity_repository):
        mock_instance = AsyncMock()
        mock_instance.generate_video = AsyncMock(side_effect=RuntimeError("always fails"))
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("virtual_streamer.video_generation.story_to_video.WanGPLTXClient", MagicMock(return_value=mock_instance)):
            with pytest.raises(RuntimeError):
                await story_input_to_video(
                    story_input=sample_story_input,
                    ltx_config=sample_ltx_config,
                    video_params=sample_video_params,
                    output_dir=output_dir,
                )


# ---------------------------------------------------------------------------
# story_input_to_video — audio routing
# ---------------------------------------------------------------------------

class TestStoryInputToVideoAudio:

    async def test_audio_forwarded_to_matching_scene_only(self, fake_audio_file, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository, mock_get_length):
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
            segment_audio_paths={0: fake_audio_file},
        )
        calls = mock_ltx_client.generate_video.call_args_list
        assert calls[0].kwargs["params"].audio_path == fake_audio_file
        assert calls[1].kwargs["params"].audio_path is None


# ---------------------------------------------------------------------------
# story_input_to_video — debug uploads
# ---------------------------------------------------------------------------

class TestStoryInputToVideoDebugUploads:

    async def test_uploads_artifacts_when_prefix_set(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository, mock_storage_client):
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
            debug_minio_prefix="test-run",
        )
        assert mock_storage_client.upload_file.call_count >= 1
        assert mock_storage_client.put_json.call_count >= 1

    async def test_no_uploads_when_prefix_absent(self, sample_story_input, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_sd_client, mock_entity_repository, mock_storage_client):
        await story_input_to_video(
            story_input=sample_story_input,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        mock_storage_client.upload_file.assert_not_called()
        mock_storage_client.put_json.assert_not_called()
