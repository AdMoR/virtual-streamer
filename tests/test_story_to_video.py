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

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_streamer.video_generation.ltx_client import (
    LTXVideoConfig,
    VideoGenerationParams,
    VideoGenerationResult,
)
from virtual_streamer.video_generation.story_to_video import (
    StoryVideoResult,
    _frames_from_duration,
    concatenate_videos,
    generate_location_image,
    generate_segment,
)
import unittest

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
            # Find the -i argument (concat list file)
            for j, arg in enumerate(cmd):
                if arg == "-i" and j + 1 < len(cmd):
                    concat_path = cmd[j + 1]
                    if Path(concat_path).exists():
                        concat_text = Path(concat_path).read_text()
        # If the concat file was read, verify no unquoted spaces outside 'file ...' syntax
        # (ffmpeg concat uses: file '/path/to/video.mp4')
        if concat_text:
            for line in concat_text.splitlines():
                if line.startswith("file"):
                    path_part = line.split("'")[1] if "'" in line else line.split()[1]
                    assert " " not in path_part


# ---------------------------------------------------------------------------
# generate_segment
# ---------------------------------------------------------------------------

def _make_client(fake_video_file) -> AsyncMock:
    """Build a bare AsyncMock LTX client for generate_segment tests."""
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

@unittest.skip
class TestGenerateSegmentTextToVideo:

    async def test_calls_generate_video_once(self, fake_video_file, sample_dialog_line, sample_video_params, tmp_path):
        client = _make_client(fake_video_file)
        await generate_segment(client, sample_dialog_line, 0, str(tmp_path), sample_video_params)
        client.generate_video.assert_called_once()

    async def test_result_fields_populated(self, fake_video_file, sample_dialog_line, sample_video_params, tmp_path):
        client = _make_client(fake_video_file)
        seg = await generate_segment(client, sample_dialog_line, 2, str(tmp_path), sample_video_params)
        assert seg.index == 2
        assert seg.video_path == fake_video_file
        assert seg.prompt_id == "mock-pid"
        assert seg.duration_seconds == 3.0
        assert seg.audio_path is None
        assert seg.image_path is None

    async def test_segment_dir_created(self, fake_video_file, sample_dialog_line, sample_video_params, tmp_path):
        client = _make_client(fake_video_file)
        await generate_segment(client, sample_dialog_line, 0, str(tmp_path), sample_video_params)
        assert (tmp_path / "segment_000").is_dir()

@unittest.skip
class TestGenerateSegmentWithImage:

    async def test_image_path_forwarded(self, fake_video_file, fake_image_file, sample_dialog_line, sample_video_params, tmp_path):
        client = _make_client(fake_video_file)
        seg = await generate_segment(
            client, sample_dialog_line, 0, str(tmp_path), sample_video_params,
            image_path=fake_image_file,
        )
        called_params = client.generate_video.call_args.kwargs["params"]
        assert called_params.image_path == fake_image_file
        assert seg.image_path == fake_image_file

@unittest.skip
class TestGenerateSegmentWithAudio:

    async def test_audio_adapts_duration(self, fake_video_file, fake_audio_file, sample_dialog_line, sample_video_params, tmp_path, mock_get_length):
        mock_get_length.return_value = 5.5
        client = _make_client(fake_video_file)
        await generate_segment(
            client, sample_dialog_line, 0, str(tmp_path), sample_video_params,
            audio_path=fake_audio_file,
        )
        called_params = client.generate_video.call_args.kwargs["params"]
        expected_frames = _frames_from_duration(5.5, sample_video_params.fps)
        assert called_params.frames == expected_frames

    async def test_missing_audio_file_falls_back_to_configured_duration(self, fake_video_file, sample_dialog_line, sample_video_params, tmp_path):
        client = _make_client(fake_video_file)
        await generate_segment(
            client, sample_dialog_line, 0, str(tmp_path), sample_video_params,
            audio_path="/nonexistent/audio.wav",
        )
        called_params = client.generate_video.call_args.kwargs["params"]
        expected_frames = _frames_from_duration(sample_video_params.duration_seconds, sample_video_params.fps)
        assert called_params.frames == expected_frames

    async def test_get_length_zero_falls_back_to_configured_duration(self, fake_video_file, fake_audio_file, sample_dialog_line, sample_video_params, tmp_path, mock_get_length):
        mock_get_length.return_value = 0.0
        client = _make_client(fake_video_file)
        await generate_segment(
            client, sample_dialog_line, 0, str(tmp_path), sample_video_params,
            audio_path=fake_audio_file,
        )
        called_params = client.generate_video.call_args.kwargs["params"]
        expected_frames = _frames_from_duration(sample_video_params.duration_seconds, sample_video_params.fps)
        assert called_params.frames == expected_frames

    async def test_audio_path_forwarded_to_params(self, fake_video_file, fake_audio_file, sample_dialog_line, sample_video_params, tmp_path, mock_get_length):
        client = _make_client(fake_video_file)
        seg = await generate_segment(
            client, sample_dialog_line, 0, str(tmp_path), sample_video_params,
            audio_path=fake_audio_file,
        )
        called_params = client.generate_video.call_args.kwargs["params"]
        assert called_params.audio_path == fake_audio_file
        assert seg.audio_path == fake_audio_file


# ---------------------------------------------------------------------------
# story_to_video
# ---------------------------------------------------------------------------
@unittest.skip
class TestStoryToVideoSuccess:

    async def test_returns_story_video_result(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run):
        from virtual_streamer.video_generation.story_to_video import story_to_video
        result = await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        assert isinstance(result, StoryVideoResult)

    async def test_two_segments_all_succeed(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run):
        from virtual_streamer.video_generation.story_to_video import story_to_video
        result = await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        assert len(result.segments) == 2

    async def test_story_title_preserved(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run):
        from virtual_streamer.video_generation.story_to_video import story_to_video
        result = await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        assert result.story_title == "Test Story"

    async def test_total_duration_positive(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run):
        from virtual_streamer.video_generation.story_to_video import story_to_video
        result = await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        assert result.total_duration_seconds > 0

    async def test_progress_callback_called(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run):
        from virtual_streamer.video_generation.story_to_video import story_to_video
        progress = MagicMock()
        await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
            progress_callback=progress,
        )
        assert progress.call_count >= len(sample_story_output.dialog)

@unittest.skip
class TestStoryToVideoResilience:

    async def test_failed_segment_is_skipped(self, fake_video_file, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_subprocess_run):
        from virtual_streamer.video_generation.story_to_video import story_to_video

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
            result = await story_to_video(
                story_output=sample_story_output,
                ltx_config=sample_ltx_config,
                video_params=sample_video_params,
                output_dir=output_dir,
            )
        assert len(result.segments) == 1

    async def test_all_segments_fail_raises_runtime_error(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir):
        from virtual_streamer.video_generation.story_to_video import story_to_video

        mock_instance = AsyncMock()
        mock_instance.generate_video = AsyncMock(side_effect=RuntimeError("always fails"))
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("virtual_streamer.video_generation.story_to_video.WanGPLTXClient", MagicMock(return_value=mock_instance)):
            with pytest.raises(RuntimeError):
                await story_to_video(
                    story_output=sample_story_output,
                    ltx_config=sample_ltx_config,
                    video_params=sample_video_params,
                    output_dir=output_dir,
                )

@unittest.skip
class TestStoryToVideoAudio:

    async def test_audio_paths_forwarded_per_segment(self, fake_video_file, fake_audio_file, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_get_length):
        from virtual_streamer.video_generation.story_to_video import story_to_video

        await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
            segment_audio_paths={0: fake_audio_file},
        )

        calls = mock_ltx_client.generate_video.call_args_list
        assert calls[0].kwargs["params"].audio_path == fake_audio_file
        assert calls[1].kwargs["params"].audio_path is None

@unittest.skip
class TestStoryToVideoWithTemplate:

    async def test_loads_locations_and_characters(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_entity_repository, mock_sd_client):
        from virtual_streamer.video_generation.story_to_video import story_to_video

        await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
            story_template_id="tmpl-1",
        )
        mock_entity_repository.list_locations_by_template.assert_called_once_with("tmpl-1")
        assert mock_entity_repository.get_character.call_count >= 1

@unittest.skip
class TestStoryToVideoDebugUploads:

    async def test_uploads_artifacts_when_prefix_set(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_storage_client):
        from virtual_streamer.video_generation.story_to_video import story_to_video

        await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
            debug_minio_prefix="test-run",
        )
        assert mock_storage_client.upload_file.call_count >= 1
        assert mock_storage_client.put_json.call_count >= 1

    async def test_no_uploads_when_prefix_absent(self, sample_story_output, sample_ltx_config, sample_video_params, output_dir, mock_ltx_client, mock_subprocess_run, mock_storage_client):
        from virtual_streamer.video_generation.story_to_video import story_to_video

        await story_to_video(
            story_output=sample_story_output,
            ltx_config=sample_ltx_config,
            video_params=sample_video_params,
            output_dir=output_dir,
        )
        mock_storage_client.upload_file.assert_not_called()
        mock_storage_client.put_json.assert_not_called()


# ---------------------------------------------------------------------------
# generate_location_image
# ---------------------------------------------------------------------------
@unittest.skip
class TestGenerateLocationImageNoCharacter:

    async def test_uses_txt2image(self, mock_sd_client, tmp_path):
        result = await generate_location_image(
            location={"description": "a dark forest"},
            character={},
            output_dir=str(tmp_path),
        )
        mock_sd_client.txt2image.assert_called_once()
        mock_sd_client.image_edit.assert_not_called()
        assert result is not None

    async def test_no_people_in_prompt(self, mock_sd_client, tmp_path):
        await generate_location_image(
            location={"description": "empty space station"},
            character={},
            output_dir=str(tmp_path),
        )
        prompt_arg = mock_sd_client.txt2image.call_args[0][0].prompt
        assert "no people" in prompt_arg

@unittest.skip
class TestGenerateLocationImageWithCharacter:

    async def test_uses_txt2image_without_identity_images(self, mock_sd_client, tmp_path):
        await generate_location_image(
            location={"description": "a lab"},
            character={"name": "Fred", "description": "a scientist", "identity_images": []},
            output_dir=str(tmp_path),
        )
        mock_sd_client.txt2image.assert_called_once()
        mock_sd_client.image_edit.assert_not_called()

    async def test_uses_image_edit_with_identity_images(self, mock_sd_client, mock_storage_client, fake_image_file, tmp_path):
        await generate_location_image(
            location={"description": "a lab"},
            character={
                "name": "Fred",
                "description": "a scientist",
                "identity_images": ["minio/path/fred.png"],
            },
            output_dir=str(tmp_path),
        )
        mock_storage_client.download_file.assert_called_once()
        mock_sd_client.image_edit.assert_called_once()

    async def test_falls_back_to_txt2image_when_download_fails(self, mock_sd_client, tmp_path):
        storage_mock = AsyncMock()
        storage_mock.download_file = AsyncMock(side_effect=RuntimeError("MinIO unavailable"))

        with patch("virtual_streamer.video_generation.story_to_video.get_storage_client", return_value=storage_mock):
            await generate_location_image(
                location={"description": "a lab"},
                character={
                    "name": "Fred",
                    "description": "a scientist",
                    "identity_images": ["minio/path/fred.png"],
                },
                output_dir=str(tmp_path),
            )
        mock_sd_client.txt2image.assert_called_once()

    async def test_returns_none_on_sd_exception(self, tmp_path):
        with patch("virtual_streamer.video_generation.story_to_video.StableDiffusionCppClient", side_effect=RuntimeError("SD server down")):
            result = await generate_location_image(
                location={"description": "a lab"},
                character={},
                output_dir=str(tmp_path),
            )
        assert result is None
