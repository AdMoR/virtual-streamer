"""
Tests for the SceneEnricher pipeline.

Covers:
- extract_evenly_spaced_frames: frame count, empty result on bad cap, partial failures
- InjectFramesCallback: correct Part objects appended to llm_request
- StoreDescriptionCallback / StoreEnrichedSceneCallback: state writes
- SceneEnrichmentInstructionProvider: prompt contains description + scene_text
- run_scene_enricher: end-to-end (mocked runner), fallback to original on failure
- scenes_to_video enrichment integration: enricher called when reference_videos provided
"""

import io
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# extract_evenly_spaced_frames
# ---------------------------------------------------------------------------

class TestExtractEvenlySpacedFrames:

    def _make_cap(self, total_frames: int = 120, read_ok: bool = True):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = float(total_frames)
        cap.read.return_value = (read_ok, fake_frame)
        return cap

    def test_returns_n_frames_on_success(self):
        cap = self._make_cap()
        with patch("virtual_streamer.agents.common.utils.cv2.VideoCapture", return_value=cap):
            from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames
            result = extract_evenly_spaced_frames("/fake/video.mp4", n=4)
        assert len(result) == 4

    def test_all_results_are_bytes(self):
        cap = self._make_cap()
        with patch("virtual_streamer.agents.common.utils.cv2.VideoCapture", return_value=cap):
            from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames
            result = extract_evenly_spaced_frames("/fake/video.mp4", n=3)
        assert all(isinstance(f, bytes) for f in result)

    def test_returns_empty_list_when_cap_not_opened(self):
        cap = MagicMock()
        cap.isOpened.return_value = False
        with patch("virtual_streamer.agents.common.utils.cv2.VideoCapture", return_value=cap):
            from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames
            result = extract_evenly_spaced_frames("/bad/path.mp4")
        assert result == []

    def test_returns_empty_list_when_total_frames_zero(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        with patch("virtual_streamer.agents.common.utils.cv2.VideoCapture", return_value=cap):
            from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames
            result = extract_evenly_spaced_frames("/zero/video.mp4")
        cap.release.assert_called_once()
        assert result == []

    def test_partial_result_when_some_frames_unreadable(self):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 120.0
        cap.read.side_effect = [
            (True, fake_frame),
            (False, None),
            (True, fake_frame),
            (True, fake_frame),
        ]
        with patch("virtual_streamer.agents.common.utils.cv2.VideoCapture", return_value=cap):
            from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames
            result = extract_evenly_spaced_frames("/fake/video.mp4", n=4)
        assert len(result) == 3

    def test_single_frame_uses_middle_index(self):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 100.0
        cap.read.return_value = (True, fake_frame)
        with patch("virtual_streamer.agents.common.utils.cv2.VideoCapture", return_value=cap):
            from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames
            result = extract_evenly_spaced_frames("/fake/video.mp4", n=1)
        assert len(result) == 1
        # index 0 arg to set() should be 50 (middle of 100 frames)
        set_calls = cap.set.call_args_list
        indices_used = [call.args[1] for call in set_calls]
        assert 50 in indices_used

    def test_cap_released_on_success(self):
        cap = self._make_cap()
        with patch("virtual_streamer.agents.common.utils.cv2.VideoCapture", return_value=cap):
            from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames
            extract_evenly_spaced_frames("/fake/video.mp4", n=2)
        cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# InjectFramesCallback
# ---------------------------------------------------------------------------

class TestInjectFramesCallback:

    def _make_llm_request(self):
        part = MagicMock()
        part.text = "some prompt"
        content = MagicMock()
        content.parts = [part]
        request = MagicMock()
        request.contents = [content]
        return request

    def _make_ctx(self, state: Optional[dict] = None):
        ctx = MagicMock()
        ctx.state = state if state is not None else {}
        return ctx

    async def test_appends_frame_parts_to_request(self):
        fake_frames = [b"FAKE_FRAME_1", b"FAKE_FRAME_2", b"FAKE_FRAME_3", b"FAKE_FRAME_4"]
        ctx = self._make_ctx({"enrichment_video_path": "/fake/video.mp4"})
        request = self._make_llm_request()

        with patch(
            "virtual_streamer.agents.scene_enricher.callback.extract_evenly_spaced_frames",
            return_value=fake_frames,
        ):
            from google.genai import types
            with patch.object(types.Part, "from_bytes", side_effect=lambda data, mime_type: MagicMock()) as mock_part:
                from virtual_streamer.agents.scene_enricher.callback import InjectFramesCallback
                cb = InjectFramesCallback(n_frames=4)
                result = await cb(ctx, request)

        assert result is None
        assert mock_part.call_count == 4

    async def test_no_op_when_no_video_path_in_state(self):
        ctx = self._make_ctx({})
        request = self._make_llm_request()
        initial_parts_len = len(request.contents[0].parts)

        from virtual_streamer.agents.scene_enricher.callback import InjectFramesCallback
        cb = InjectFramesCallback()
        result = await cb(ctx, request)

        assert result is None
        assert len(request.contents[0].parts) == initial_parts_len

    async def test_no_op_when_frames_empty(self):
        ctx = self._make_ctx({"enrichment_video_path": "/bad/video.mp4"})
        request = self._make_llm_request()
        initial_parts_len = len(request.contents[0].parts)

        with patch(
            "virtual_streamer.agents.scene_enricher.callback.extract_evenly_spaced_frames",
            return_value=[],
        ):
            from virtual_streamer.agents.scene_enricher.callback import InjectFramesCallback
            cb = InjectFramesCallback()
            result = await cb(ctx, request)

        assert result is None
        assert len(request.contents[0].parts) == initial_parts_len


# ---------------------------------------------------------------------------
# StoreDescriptionCallback / StoreEnrichedSceneCallback
# ---------------------------------------------------------------------------

class TestStoreCallbacks:

    def _make_llm_response(self, text: str):
        part = MagicMock()
        part.text = text
        content = MagicMock()
        content.parts = [part]
        response = MagicMock()
        response.content = content
        return response

    def _make_ctx(self):
        ctx = MagicMock()
        ctx.state = {}
        return ctx

    async def test_store_description_writes_to_state(self):
        from virtual_streamer.agents.scene_enricher.callback import StoreDescriptionCallback
        ctx = self._make_ctx()
        response = self._make_llm_response("The actor walks forward.")
        cb = StoreDescriptionCallback()
        await cb(ctx, response)
        assert ctx.state["enrichment_video_description"] == "The actor walks forward."

    async def test_store_enriched_scene_writes_to_state(self):
        from virtual_streamer.agents.scene_enricher.callback import StoreEnrichedSceneCallback
        ctx = self._make_ctx()
        response = self._make_llm_response("A scientist strides purposefully.")
        cb = StoreEnrichedSceneCallback()
        await cb(ctx, response)
        assert ctx.state["enriched_scene"] == "A scientist strides purposefully."

    async def test_store_description_handles_empty_response(self):
        from virtual_streamer.agents.scene_enricher.callback import StoreDescriptionCallback
        ctx = self._make_ctx()
        response = MagicMock()
        response.content = None
        cb = StoreDescriptionCallback()
        await cb(ctx, response)
        assert ctx.state["enrichment_video_description"] == ""


# ---------------------------------------------------------------------------
# SceneEnrichmentInstructionProvider
# ---------------------------------------------------------------------------

class TestSceneEnrichmentInstructionProvider:

    async def test_prompt_contains_description_and_scene(self):
        from virtual_streamer.agents.scene_enricher.prompt import SceneEnrichmentInstructionProvider
        provider = SceneEnrichmentInstructionProvider()
        ctx = MagicMock()
        ctx.state = {
            "enrichment_video_description": "Someone opens a cabinet.",
            "enrichment_scene_text": "A man in a garden talks to the camera.",
        }
        prompt = await provider(ctx)
        assert "Someone opens a cabinet." in prompt
        assert "A man in a garden talks to the camera." in prompt

    async def test_prompt_uses_empty_strings_when_keys_missing(self):
        from virtual_streamer.agents.scene_enricher.prompt import SceneEnrichmentInstructionProvider
        provider = SceneEnrichmentInstructionProvider()
        ctx = MagicMock()
        ctx.state = {}
        prompt = await provider(ctx)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    async def test_prompt_contains_scene_tag(self):
        from virtual_streamer.agents.scene_enricher.prompt import SceneEnrichmentInstructionProvider
        provider = SceneEnrichmentInstructionProvider()
        ctx = MagicMock()
        ctx.state = {"enrichment_video_description": "x", "enrichment_scene_text": "y"}
        prompt = await provider(ctx)
        assert "[SCENE]" in prompt


# ---------------------------------------------------------------------------
# run_scene_enricher (end-to-end with mocked runner)
# ---------------------------------------------------------------------------

class TestRunSceneEnricher:

    async def test_returns_enriched_text_from_state(self):
        mock_session_after = MagicMock()
        mock_session_after.state = {"enriched_scene": "Enriched version of the scene."}

        mock_session_before = MagicMock()
        mock_session_before.id = "run_test123"

        mock_session_service = AsyncMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session_before)
        mock_session_service.get_session = AsyncMock(return_value=mock_session_after)

        async def _fake_run_async(**kwargs):
            return
            yield  # make it an async generator

        mock_runner = MagicMock()
        mock_runner.run_async = MagicMock(return_value=_fake_run_async())

        with patch("virtual_streamer.agents.scene_enricher.agent.InMemorySessionService",
                   return_value=mock_session_service):
            with patch("virtual_streamer.agents.scene_enricher.agent.Runner",
                       return_value=mock_runner):
                with patch("virtual_streamer.agents.scene_enricher.agent.get_scene_enricher_pipeline",
                           return_value=MagicMock()):
                    from virtual_streamer.agents.scene_enricher.agent import run_scene_enricher
                    result = await run_scene_enricher("/fake/video.mp4", "original scene text")

        assert result == "Enriched version of the scene."

    async def test_returns_original_when_enriched_scene_missing(self):
        mock_session_after = MagicMock()
        mock_session_after.state = {}

        mock_session_before = MagicMock()
        mock_session_before.id = "run_test456"

        mock_session_service = AsyncMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session_before)
        mock_session_service.get_session = AsyncMock(return_value=mock_session_after)

        async def _fake_run_async(**kwargs):
            return
            yield

        mock_runner = MagicMock()
        mock_runner.run_async = MagicMock(return_value=_fake_run_async())

        with patch("virtual_streamer.agents.scene_enricher.agent.InMemorySessionService",
                   return_value=mock_session_service):
            with patch("virtual_streamer.agents.scene_enricher.agent.Runner",
                       return_value=mock_runner):
                with patch("virtual_streamer.agents.scene_enricher.agent.get_scene_enricher_pipeline",
                           return_value=MagicMock()):
                    from virtual_streamer.agents.scene_enricher.agent import run_scene_enricher
                    result = await run_scene_enricher("/fake/video.mp4", "fallback scene")

        assert result == "fallback scene"

    async def test_returns_original_on_runner_exception(self):
        mock_session_before = MagicMock()
        mock_session_before.id = "run_err"

        mock_session_service = AsyncMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session_before)

        async def _failing_run(**kwargs):
            raise RuntimeError("LLM error")
            yield

        mock_runner = MagicMock()
        mock_runner.run_async = MagicMock(return_value=_failing_run())

        with patch("virtual_streamer.agents.scene_enricher.agent.InMemorySessionService",
                   return_value=mock_session_service):
            with patch("virtual_streamer.agents.scene_enricher.agent.Runner",
                       return_value=mock_runner):
                with patch("virtual_streamer.agents.scene_enricher.agent.get_scene_enricher_pipeline",
                           return_value=MagicMock()):
                    from virtual_streamer.agents.scene_enricher.agent import run_scene_enricher
                    result = await run_scene_enricher("/fake/video.mp4", "original text")

        assert result == "original text"


# ---------------------------------------------------------------------------
# scenes_to_video integration: enrichment called / skipped
# ---------------------------------------------------------------------------

class TestScenestoVideoEnrichment:

    async def test_enrichment_called_for_mapped_scene(
        self,
        sample_ltx_config,
        sample_video_params,
        output_dir,
        mock_ltx_client,
        mock_sd_client,
        mock_entity_repository,
        mock_get_length,
        mock_subprocess_run,
        fake_video_file,
    ):
        from virtual_streamer.agents.story_pipeline.schema import DetailedScene
        from virtual_streamer.image_generation.models import FluxPrompt, Camera

        scene = DetailedScene(
            ltx_prompt="original prompt",
            scene_visual_description=FluxPrompt(
                scene="lab",
                subjects=[],
                lighting="soft",
                camera=Camera(angle="eye level", distance="medium shot"),
            ),
        )

        with patch(
            "virtual_streamer.video_generation.story_to_video.run_scene_enricher",
            new_callable=AsyncMock,
            return_value="enriched prompt",
        ) as mock_enrich:
            from virtual_streamer.video_generation.story_to_video import scenes_to_video
            await scenes_to_video(
                scenes=[scene],
                ltx_config=sample_ltx_config,
                video_params=sample_video_params,
                output_dir=output_dir,
                reference_videos={0: fake_video_file},
            )

        mock_enrich.assert_called_once_with(fake_video_file, "original prompt")

    async def test_enrichment_skipped_when_no_reference_videos(
        self,
        sample_ltx_config,
        sample_video_params,
        output_dir,
        mock_ltx_client,
        mock_sd_client,
        mock_entity_repository,
        mock_get_length,
        mock_subprocess_run,
    ):
        from virtual_streamer.agents.story_pipeline.schema import DetailedScene
        from virtual_streamer.image_generation.models import FluxPrompt, Camera

        scene = DetailedScene(
            ltx_prompt="original prompt",
            scene_visual_description=FluxPrompt(
                scene="lab",
                subjects=[],
                lighting="soft",
                camera=Camera(angle="eye level", distance="medium shot"),
            ),
        )

        with patch(
            "virtual_streamer.video_generation.story_to_video.run_scene_enricher",
            new_callable=AsyncMock,
        ) as mock_enrich:
            from virtual_streamer.video_generation.story_to_video import scenes_to_video
            await scenes_to_video(
                scenes=[scene],
                ltx_config=sample_ltx_config,
                video_params=sample_video_params,
                output_dir=output_dir,
                reference_videos=None,
            )

        mock_enrich.assert_not_called()

    async def test_enrichment_skipped_when_video_path_missing(
        self,
        sample_ltx_config,
        sample_video_params,
        output_dir,
        mock_ltx_client,
        mock_sd_client,
        mock_entity_repository,
        mock_get_length,
        mock_subprocess_run,
    ):
        from virtual_streamer.agents.story_pipeline.schema import DetailedScene
        from virtual_streamer.image_generation.models import FluxPrompt, Camera

        scene = DetailedScene(
            ltx_prompt="original prompt",
            scene_visual_description=FluxPrompt(
                scene="lab",
                subjects=[],
                lighting="soft",
                camera=Camera(angle="eye level", distance="medium shot"),
            ),
        )

        with patch(
            "virtual_streamer.video_generation.story_to_video.run_scene_enricher",
            new_callable=AsyncMock,
        ) as mock_enrich:
            from virtual_streamer.video_generation.story_to_video import scenes_to_video
            await scenes_to_video(
                scenes=[scene],
                ltx_config=sample_ltx_config,
                video_params=sample_video_params,
                output_dir=output_dir,
                reference_videos={0: "/nonexistent/video.mp4"},
            )

        mock_enrich.assert_not_called()
