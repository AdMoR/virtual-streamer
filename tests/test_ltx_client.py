"""
Tests for ltx_client.py — VideoGenerationParams validation and _build_settings
serialisation logic (pure Pydantic + dict, no network I/O).
"""

import pytest
from unittest.mock import patch

from virtual_streamer.video_generation.ltx_client import (
    VideoGenerationParams,
    WanGPLTXClient,
    LTXVideoConfig,
    _DEFAULTS,
    _PATH_FIELDS,
    _CONVENIENCE_FIELDS,
)
from virtual_streamer.video_generation.story_to_video import _frames_from_duration


# ---------------------------------------------------------------------------
# VideoGenerationParams — field defaults and validation
# ---------------------------------------------------------------------------

class TestVideoGenerationParams:

    def test_resolution_default(self):
        params = VideoGenerationParams(prompt="test")
        assert params.resolution == "1280x720"

    def test_resolution_set_explicitly(self):
        params = VideoGenerationParams(prompt="test", resolution="832x480")
        assert params.resolution == "832x480"
        assert params.effective_resolution == "832x480"

    def test_backward_compat_width_height(self):
        params = VideoGenerationParams(prompt="test", resolution="832x480")
        assert params.width == 832
        assert params.height == 480

    def test_video_length_default(self):
        params = VideoGenerationParams(prompt="test")
        assert params.video_length == 97

    def test_video_length_explicit(self):
        params = VideoGenerationParams(prompt="test", video_length=121)
        assert params.video_length == 121

    def test_duration_seconds_computes_video_length(self):
        # 4.0s * 24fps = 96 raw → n=round(95/8)=12 → 8*12+1=97
        params = VideoGenerationParams(prompt="test", duration_seconds=4.0, fps=24)
        assert params.video_length == 97

    def test_duration_seconds_minimum_nine_frames(self):
        params = VideoGenerationParams(prompt="test", duration_seconds=0.1, fps=24)
        assert params.video_length == 9

    def test_duration_seconds_5s_24fps(self):
        # 5.0 * 24 = 120 → n=round(119/8)=15 → 121
        params = VideoGenerationParams(prompt="test", duration_seconds=5.0, fps=24)
        assert params.video_length == 121

    def test_guidance_scale_default(self):
        params = VideoGenerationParams(prompt="test")
        assert params.guidance_scale == 1.0

    def test_num_inference_steps_default(self):
        params = VideoGenerationParams(prompt="test")
        assert params.num_inference_steps == 8

    def test_backward_compat_steps_property(self):
        params = VideoGenerationParams(prompt="test", num_inference_steps=30)
        assert params.steps == 30

    def test_video_prompt_type_default_empty(self):
        params = VideoGenerationParams(prompt="test")
        assert params.video_prompt_type == ""

    def test_image_prompt_type_default_empty(self):
        params = VideoGenerationParams(prompt="test")
        assert params.image_prompt_type == ""

    def test_audio_prompt_type_default_empty(self):
        params = VideoGenerationParams(prompt="test")
        assert params.audio_prompt_type == ""

    def test_denoising_strength_out_of_range(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            VideoGenerationParams(prompt="test", denoising_strength=1.5)

    def test_file_path_fields_default_none_or_empty(self):
        params = VideoGenerationParams(prompt="test")
        assert params.image_start is None
        assert params.image_end is None
        assert params.audio_guide is None
        assert params.video_guide is None
        assert params.video_mask is None
        assert params.image_refs == []
        assert params.keyframes == []

    def test_optional_advanced_fields_default_none(self):
        params = VideoGenerationParams(prompt="test")
        assert params.guidance_phases is None
        assert params.sample_solver is None
        assert params.alt_guidance_scale is None
        assert params.alt_scale is None
        assert params.perturbation_switch is None
        assert params.perturbation_layers is None
        assert params.NAG_scale is None
        assert params.keep_frames_video_guide is None
        assert params.masking_strength is None
        assert params.mask_expand is None
        assert params.sliding_window_size is None

    def test_actual_duration(self):
        params = VideoGenerationParams(prompt="test", video_length=97, fps=24)
        assert abs(params.actual_duration - 97 / 24) < 0.01

    def test_from_preset_fast(self):
        params = VideoGenerationParams.from_preset("fast", prompt="test")
        assert params.model_type == "ltx2_22B_distilled_1_1"
        assert params.num_inference_steps == 8
        assert params.guidance_scale == 1.0

    def test_from_preset_quality(self):
        params = VideoGenerationParams.from_preset("quality", prompt="test")
        assert params.model_type == "ltx2_22B"
        assert params.num_inference_steps == 30

    def test_from_preset_high_quality(self):
        params = VideoGenerationParams.from_preset("high_quality", prompt="test")
        assert params.model_type == "ltx2_22B_pure_dev"
        assert params.num_inference_steps == 50


# ---------------------------------------------------------------------------
# WanGPLTXClient._build_settings — payload serialisation
# ---------------------------------------------------------------------------

class TestBuildSettings:
    """Unit-test _build_settings without any network I/O."""

    def _client(self) -> WanGPLTXClient:
        return WanGPLTXClient(LTXVideoConfig(server_url="http://localhost:9999"))

    def _base_params(self, **overrides) -> VideoGenerationParams:
        defaults: dict = {
            "prompt": "a woman in a park",
            "model_type": "ltx2_22B_distilled_1_1",
            "resolution": "832x480",
            "video_length": 97,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "seed": 42,
        }
        defaults.update(overrides)
        return VideoGenerationParams(**defaults)

    # ── Core fields always present ────────────────────────────────────────────

    def test_core_fields_always_in_settings(self):
        client = self._client()
        params = self._base_params()
        settings = client._build_settings(params)
        for key in ("model_type", "prompt", "negative_prompt", "resolution",
                    "video_length", "num_inference_steps", "guidance_scale",
                    "flow_shift", "seed"):
            assert key in settings, f"Expected {key!r} in settings"

    def test_resolution_in_settings(self):
        client = self._client()
        params = self._base_params(resolution="832x480")
        settings = client._build_settings(params)
        assert settings["resolution"] == "832x480"

    def test_resolution_override_applied(self):
        client = self._client()
        params = self._base_params(resolution="1280x720")
        settings = client._build_settings(params, resolution_override="480x270")
        assert settings["resolution"] == "480x270"

    # ── File path fields excluded ─────────────────────────────────────────────

    def test_path_fields_not_in_settings(self):
        client = self._client()
        params = self._base_params(
            image_start="/tmp/img.png",
            audio_guide="/tmp/audio.wav",
            video_guide="/tmp/video.mp4",
        )
        settings = client._build_settings(params)
        for path_field in _PATH_FIELDS:
            assert path_field not in settings, f"Path field {path_field!r} should be excluded"

    def test_convenience_fields_not_in_settings(self):
        client = self._client()
        params = self._base_params(duration_seconds=4.0, fps=24)
        settings = client._build_settings(params)
        for conv_field in _CONVENIENCE_FIELDS:
            assert conv_field not in settings, f"Convenience field {conv_field!r} should be excluded"

    # ── Empty string flags excluded ───────────────────────────────────────────

    def test_empty_video_prompt_type_excluded(self):
        client = self._client()
        params = self._base_params(video_prompt_type="")
        settings = client._build_settings(params)
        assert "video_prompt_type" not in settings

    def test_empty_image_prompt_type_excluded(self):
        client = self._client()
        params = self._base_params(image_prompt_type="")
        settings = client._build_settings(params)
        assert "image_prompt_type" not in settings

    def test_empty_audio_prompt_type_excluded(self):
        client = self._client()
        params = self._base_params(audio_prompt_type="")
        settings = client._build_settings(params)
        assert "audio_prompt_type" not in settings

    def test_set_video_prompt_type_included(self):
        client = self._client()
        params = self._base_params(video_prompt_type="DVG")
        settings = client._build_settings(params)
        assert settings["video_prompt_type"] == "DVG"

    def test_set_image_prompt_type_included(self):
        client = self._client()
        params = self._base_params(image_prompt_type="S")
        settings = client._build_settings(params)
        assert settings["image_prompt_type"] == "S"

    def test_set_audio_prompt_type_included(self):
        client = self._client()
        params = self._base_params(audio_prompt_type="A")
        settings = client._build_settings(params)
        assert settings["audio_prompt_type"] == "A"

    # ── None optional fields excluded ─────────────────────────────────────────

    def test_none_optional_fields_excluded(self):
        client = self._client()
        params = self._base_params()  # all Optional fields are None
        settings = client._build_settings(params)
        for field in ("guidance_phases", "sample_solver", "perturbation_switch",
                      "NAG_scale", "keep_frames_video_guide", "masking_strength",
                      "mask_expand", "sliding_window_size", "remove_background_images_ref"):
            assert field not in settings, f"Optional field {field!r} should be excluded when None"

    def test_sample_solver_included_when_set(self):
        client = self._client()
        params = self._base_params(sample_solver="distilled_8_steps", guidance_phases=2)
        settings = client._build_settings(params)
        assert settings["sample_solver"] == "distilled_8_steps"
        assert settings["guidance_phases"] == 2

    def test_perturbation_params_included(self):
        client = self._client()
        params = self._base_params(perturbation_switch=2, perturbation_layers=[28])
        settings = client._build_settings(params)
        assert settings["perturbation_switch"] == 2
        assert settings["perturbation_layers"] == [28]

    def test_NAG_scale_included(self):
        client = self._client()
        params = self._base_params(NAG_scale=1.5)
        settings = client._build_settings(params)
        assert settings["NAG_scale"] == 1.5

    def test_keep_frames_video_guide_passthrough(self):
        client = self._client()
        params = self._base_params(keep_frames_video_guide="17:-1")
        settings = client._build_settings(params)
        assert settings["keep_frames_video_guide"] == "17:-1"

    def test_sliding_window_params_included(self):
        client = self._client()
        params = self._base_params(
            sliding_window_size=97,
            sliding_window_overlap=17,
            sliding_window_color_correction_strength=0.5,
        )
        settings = client._build_settings(params)
        assert settings["sliding_window_size"] == 97
        assert settings["sliding_window_overlap"] == 17
        assert settings["sliding_window_color_correction_strength"] == 0.5

    # ── Empty LoRA list excluded ──────────────────────────────────────────────

    def test_empty_loras_excluded(self):
        client = self._client()
        params = self._base_params(activated_loras=[], loras_multipliers="")
        settings = client._build_settings(params)
        assert "activated_loras" not in settings
        assert "loras_multipliers" not in settings

    def test_loras_included_when_set(self):
        client = self._client()
        params = self._base_params(
            activated_loras=["my-style.safetensors"],
            loras_multipliers="0.8",
        )
        settings = client._build_settings(params)
        assert settings["activated_loras"] == ["my-style.safetensors"]
        assert settings["loras_multipliers"] == "0.8"

    # ── File ref injection ────────────────────────────────────────────────────

    def test_image_start_ref_injected(self):
        client = self._client()
        params = self._base_params(image_prompt_type="S")
        settings = client._build_settings(params, image_start_id="abc123")
        assert settings["image_start"] == "file:abc123"

    def test_audio_guide_ref_injected(self):
        client = self._client()
        params = self._base_params(audio_prompt_type="A")
        settings = client._build_settings(params, audio_guide_id="audio_xyz")
        assert settings["audio_guide"] == "file:audio_xyz"

    def test_video_guide_ref_injected(self):
        client = self._client()
        params = self._base_params(video_prompt_type="DVG")
        settings = client._build_settings(params, video_guide_id="vid_xyz")
        assert settings["video_guide"] == "file:vid_xyz"

    def test_image_refs_injected(self):
        client = self._client()
        params = self._base_params(video_prompt_type="I")
        settings = client._build_settings(params, image_ref_ids=["ref_a", "ref_b"])
        assert settings["image_refs"] == ["file:ref_a", "file:ref_b"]

    def test_keyframes_refs_injected(self):
        client = self._client()
        params = self._base_params(
            model_type="ltx2_22B_keyframe",
            keyframes=[["frame0.png", 0, 1.0], ["frame60.png", 60, 1.0]],
        )
        settings = client._build_settings(
            params,
            keyframe_ids=["fid_0", "fid_60"],
        )
        assert settings["keyframes"] == [
            ["file:fid_0", 0, 1.0],
            ["file:fid_60", 60, 1.0],
        ]

    def test_video_mask_ref_injected(self):
        client = self._client()
        params = self._base_params()
        settings = client._build_settings(params, video_mask_id="mask_abc")
        assert settings["video_mask"] == "file:mask_abc"

    def test_remove_background_included_when_set(self):
        client = self._client()
        params = self._base_params(remove_background_images_ref=0)
        settings = client._build_settings(params)
        assert settings["remove_background_images_ref"] == 0

    # ── Talking-head / audio with loras ──────────────────────────────────────

    def test_talking_head_mode_settings(self):
        """A1O audio mode with ID-LoRA and image start frame."""
        client = self._client()
        params = self._base_params(
            image_prompt_type="S",
            audio_prompt_type="A1O",
            audio_scale=1.0,
            audio_guidance_scale=5.0,
            guidance_scale=2.0,
            flow_shift=1.0,
            guidance_phases=2,
            sample_solver="distilled_8_steps",
            activated_loras=["id-lora-celebvhq-ltx2.3.safetensors"],
            loras_multipliers="1.0",
        )
        settings = client._build_settings(
            params,
            image_start_id="img_id",
            audio_guide_id="audio_id",
        )
        assert settings["audio_prompt_type"] == "A1O"
        assert settings["image_prompt_type"] == "S"
        assert settings["image_start"] == "file:img_id"
        assert settings["audio_guide"] == "file:audio_id"
        assert settings["activated_loras"] == ["id-lora-celebvhq-ltx2.3.safetensors"]
        assert settings["loras_multipliers"] == "1.0"
        assert settings["guidance_phases"] == 2
        assert settings["sample_solver"] == "distilled_8_steps"


# ---------------------------------------------------------------------------
# _submit_job endpoint
# ---------------------------------------------------------------------------

class TestSubmitJobEndpoint:
    """Verify the client posts to /jobs/raw (not /jobs)."""

    def test_submit_uses_jobs_raw(self):
        client = WanGPLTXClient(LTXVideoConfig(server_url="http://mock-server:8082"))
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 202
            mock_post.return_value.json.return_value = {
                "job_id": "test_job_id",
                "queue_position": 0,
            }
            job_id, queue_pos = client._submit_job({"model_type": "ltx2_22B_distilled_1_1"})
        called_url = mock_post.call_args[0][0]
        assert called_url.endswith("/jobs/raw"), (
            f"Expected endpoint to be /jobs/raw, got: {called_url}"
        )
        assert job_id == "test_job_id"


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
