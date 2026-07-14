"""
Tests for virtual_streamer/video_generation/strategies/.

Covers factory precedence (select_strategy) and each strategy's
VideoGenerationParams output. ffmpeg is mocked for the reference-sheet
strategy — no real video encoding happens.
"""

from unittest.mock import MagicMock, patch

import pytest

from virtual_streamer.video_generation.strategies import (
    ConditioningContext,
    ImageConditioningStrategy,
    ReferenceSheetStrategy,
    TalkingHeadStrategy,
    select_strategy,
)
from virtual_streamer.video_generation.strategies.reference_sheet import (
    REFERENCE_SHEET_LORA,
    REFERENCE_SHEET_MODEL_TYPE,
)
from virtual_streamer.video_generation.strategies.talking_head import TALKING_HEAD_LORA


def _make_ctx(sample_scene_input, sample_video_params, output_dir, **kwargs):
    return ConditioningContext(
        scene_input=sample_scene_input,
        video_params=sample_video_params,
        output_dir=output_dir,
        **kwargs,
    )


class TestSelectStrategy:
    def test_no_conditioning_falls_back_to_i2v(
        self, sample_scene_input, sample_video_params, output_dir
    ):
        ctx = _make_ctx(sample_scene_input, sample_video_params, output_dir)
        assert isinstance(select_strategy(ctx), ImageConditioningStrategy)

    def test_image_only_uses_i2v(
        self, sample_scene_input, sample_video_params, output_dir, fake_image_file
    ):
        ctx = _make_ctx(
            sample_scene_input, sample_video_params, output_dir, image_path=fake_image_file
        )
        assert isinstance(select_strategy(ctx), ImageConditioningStrategy)

    def test_audio_selects_talking_head(
        self, sample_scene_input, sample_video_params, output_dir, fake_audio_file
    ):
        ctx = _make_ctx(
            sample_scene_input, sample_video_params, output_dir, audio_path=fake_audio_file
        )
        assert isinstance(select_strategy(ctx), TalkingHeadStrategy)

    def test_missing_audio_file_does_not_select_talking_head(
        self, sample_scene_input, sample_video_params, output_dir
    ):
        ctx = _make_ctx(
            sample_scene_input, sample_video_params, output_dir,
            audio_path="/nonexistent/audio.wav",
        )
        assert isinstance(select_strategy(ctx), ImageConditioningStrategy)

    def test_reference_sheet_selects_reference_sheet_strategy(
        self, sample_scene_input, sample_video_params, output_dir, fake_image_file
    ):
        sample_scene_input.reference_sheet_path = fake_image_file
        ctx = _make_ctx(sample_scene_input, sample_video_params, output_dir)
        assert isinstance(select_strategy(ctx), ReferenceSheetStrategy)

    def test_audio_shadows_reference_sheet_with_warning(
        self, sample_scene_input, sample_video_params, output_dir,
        fake_audio_file, fake_image_file, caplog,
    ):
        sample_scene_input.reference_sheet_path = fake_image_file
        ctx = _make_ctx(
            sample_scene_input, sample_video_params, output_dir, audio_path=fake_audio_file
        )
        with caplog.at_level("WARNING"):
            strategy = select_strategy(ctx)
        assert isinstance(strategy, TalkingHeadStrategy)
        assert any("shadowed" in rec.message for rec in caplog.records)


class TestTalkingHeadStrategy:
    @pytest.mark.asyncio
    async def test_build_params(
        self, sample_scene_input, sample_video_params, output_dir,
        fake_audio_file, fake_image_file,
    ):
        ctx = _make_ctx(
            sample_scene_input, sample_video_params, output_dir,
            audio_path=fake_audio_file, image_path=fake_image_file,
        )
        params = await TalkingHeadStrategy().build_params(ctx)
        assert params.audio_prompt_type == "A1O"
        assert params.image_prompt_type == "S"
        assert params.audio_guide == fake_audio_file
        assert params.image_start == fake_image_file
        assert TALKING_HEAD_LORA in params.activated_loras
        assert params.video_length % 8 == 1


class TestImageConditioningStrategy:
    @pytest.mark.asyncio
    async def test_i2v_params(
        self, sample_scene_input, sample_video_params, output_dir, fake_image_file
    ):
        ctx = _make_ctx(
            sample_scene_input, sample_video_params, output_dir, image_path=fake_image_file
        )
        params = await ImageConditioningStrategy().build_params(ctx)
        assert params.image_start == fake_image_file
        assert params.image_prompt_type == "S"
        assert params.prompt == sample_scene_input.ltx_prompt
        assert params.video_length % 8 == 1

    @pytest.mark.asyncio
    async def test_t2v_params(self, sample_scene_input, sample_video_params, output_dir):
        ctx = _make_ctx(sample_scene_input, sample_video_params, output_dir)
        params = await ImageConditioningStrategy().build_params(ctx)
        assert params.image_start is None
        assert params.image_prompt_type == ""


class TestReferenceSheetStrategy:
    @pytest.mark.asyncio
    async def test_build_params(
        self, sample_scene_input, sample_video_params, output_dir, fake_image_file
    ):
        sample_scene_input.reference_sheet_path = fake_image_file
        sample_scene_input.reference_sheet_description = "### Reference Sheet Description ..."
        ctx = _make_ctx(sample_scene_input, sample_video_params, output_dir)

        def _fake_ffmpeg(cmd, capture_output, text):
            # cmd ends with the output video path — create it so the strategy's
            # existence check passes
            import os
            os.makedirs(os.path.dirname(cmd[-1]), exist_ok=True)
            with open(cmd[-1], "wb") as f:
                f.write(b"\x00")
            return MagicMock(returncode=0, stderr="")

        with patch(
            "virtual_streamer.video_generation.strategies.reference_sheet.subprocess.run",
            side_effect=_fake_ffmpeg,
        ) as mock_run:
            params = await ReferenceSheetStrategy().build_params(ctx)

        assert params.model_type == REFERENCE_SHEET_MODEL_TYPE
        assert REFERENCE_SHEET_LORA in params.activated_loras
        assert params.video_prompt_type == "VG"
        assert params.video_guide.endswith("reference_sheet.mp4")
        # Sheet description is prepended before the target prompt
        assert params.prompt.startswith("### Reference Sheet Description")
        assert sample_scene_input.ltx_prompt in params.prompt
        assert params.video_length % 8 == 1
        # ffmpeg scales the sheet to the output resolution (downscale factor 1)
        ffmpeg_cmd = mock_run.call_args.args[0]
        assert any("scale=1280:720" in arg for arg in ffmpeg_cmd)

    @pytest.mark.asyncio
    async def test_ffmpeg_failure_raises(
        self, sample_scene_input, sample_video_params, output_dir, fake_image_file
    ):
        sample_scene_input.reference_sheet_path = fake_image_file
        ctx = _make_ctx(sample_scene_input, sample_video_params, output_dir)
        with patch(
            "virtual_streamer.video_generation.strategies.reference_sheet.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="boom"),
        ):
            with pytest.raises(RuntimeError, match="reference sheet"):
                await ReferenceSheetStrategy().build_params(ctx)
