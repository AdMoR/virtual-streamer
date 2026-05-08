"""
Tests for ltx_client.py — VideoGenerationParams validation (pure Pydantic, no I/O).
"""

import pytest

from virtual_streamer.video_generation.ltx_client import VideoGenerationParams, _DEFAULTS
from virtual_streamer.video_generation.story_to_video import _frames_from_duration


# ---------------------------------------------------------------------------
# VideoGenerationParams
# ---------------------------------------------------------------------------

class TestVideoGenerationParams:

    def test_resolution_derived_from_width_height(self):
        params = VideoGenerationParams(prompt="test", width=1280, height=720)
        assert params.effective_resolution == "1280x720"

    def test_explicit_resolution_wins_over_width_height(self):
        params = VideoGenerationParams(prompt="test", resolution="640x480", width=1280, height=720)
        assert params.effective_resolution == "640x480"

    def test_default_resolution_string_empty(self):
        params = VideoGenerationParams(prompt="test")
        assert params.resolution == ""
        assert "x" in params.effective_resolution

    def test_cfg_scale_alias_overrides_guidance_scale(self):
        # cfg_scale differs from the default (3.0), so guidance_scale is updated
        params = VideoGenerationParams(prompt="test", cfg_scale=7.0)
        assert params.guidance_scale == 7.0

    def test_cfg_scale_at_default_leaves_guidance_scale_unchanged(self):
        # cfg_scale == default → validator does NOT overwrite explicit guidance_scale
        params = VideoGenerationParams(
            prompt="test",
            cfg_scale=_DEFAULTS["guidance_scale"],
            guidance_scale=5.0,
        )
        assert params.guidance_scale == 5.0

    def test_effective_fps_from_fps_field(self):
        params = VideoGenerationParams(prompt="test", fps=30)
        assert params.effective_fps == "30"

    def test_force_fps_overrides_fps_field(self):
        params = VideoGenerationParams(prompt="test", fps=24, force_fps="50")
        assert params.effective_fps == "50"

    def test_explicit_frames_returned_directly(self):
        params = VideoGenerationParams(prompt="test", frames=97)
        assert params.effective_frames == 97

    def test_frames_derived_from_duration_and_fps(self):
        # 4.0s * 24fps = 96 raw → n=round(95/8)=12 → max(8*12+1, 9)=97
        params = VideoGenerationParams(prompt="test", duration_seconds=4.0, fps=24, frames=0)
        assert params.effective_frames == 97

    def test_frames_minimum_nine(self):
        # Very short duration → minimum 9 frames
        params = VideoGenerationParams(prompt="test", duration_seconds=0.1, fps=24, frames=0)
        assert params.effective_frames == 9

    def test_actual_duration_computed(self):
        params = VideoGenerationParams(prompt="test", frames=97, fps=24)
        assert abs(params.actual_duration - 97 / 24) < 0.01

    def test_v2v_fields_default(self):
        params = VideoGenerationParams(prompt="test")
        assert params.video_path is None
        assert params.denoising_strength == 0.7
        assert params.video_prompt_type == "DVG"

    def test_v2v_video_path_set(self):
        params = VideoGenerationParams(prompt="test", video_path="/tmp/source.mp4")
        assert params.video_path == "/tmp/source.mp4"

    def test_v2v_denoising_strength_range(self):
        params = VideoGenerationParams(prompt="test", video_path="/tmp/v.mp4", denoising_strength=0.4)
        assert params.denoising_strength == 0.4

    def test_v2v_denoising_strength_out_of_range(self):
        import pytest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            VideoGenerationParams(prompt="test", denoising_strength=1.5)


# ---------------------------------------------------------------------------
# _frames_from_duration (story_to_video helper)
# ---------------------------------------------------------------------------

class TestFramesFromDuration:

    def test_snaps_to_8n_plus_1(self):
        # 4.0s * 24fps = 96 raw → n=max(round(95/8), 1)=12 → 8*12+1=97
        assert _frames_from_duration(4.0, 24) == 97

    def test_minimum_nine_frames(self):
        # 0.1s * 24fps = 2 raw → n=max(round(1/8), 1)=1 → 9
        assert _frames_from_duration(0.1, 24) == 9

    def test_large_duration(self):
        # 30.0s * 24fps = 720 raw → n=max(round(719/8), 1)=90 → 721
        assert _frames_from_duration(30.0, 24) == 721

    @pytest.mark.parametrize("duration,fps,expected", [
        (1.0,  24,  25),   # raw=24 → n=max(round(23/8),1)=3 → 25
        (2.0,  24,  49),   # raw=48 → n=max(round(47/8),1)=6 → 49
        (5.0,  24,  121),  # raw=120 → n=max(round(119/8),1)=15 → 121
        (4.0,  30,  121),  # raw=120 → n=max(round(119/8),1)=15 → 121
    ])
    def test_various_durations(self, duration, fps, expected):
        assert _frames_from_duration(duration, fps) == expected
