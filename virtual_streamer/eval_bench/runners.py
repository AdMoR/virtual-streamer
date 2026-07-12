"""
Per-model-kind runners for the evaluation bench.

Each runner executes ONE case against ONE model and returns a RunnerOutput:
either a local artifact file (image/video/audio, uploaded to MinIO by the
harness) or a JSON-serialisable output (LLM agents).

Registry:
    run_case = RUNNERS[model_kind]
    output = await run_case(case_params, model_id, model_config, output_dir)

model_id meaning per kind:
    image      — client implementation: "stable_cpp" (default) or "openai"
    video      — quality preset: "fast", "quality", "high_quality"
    tts        — character_id whose voice sample to clone ("default" = no clone)
    llm_agent  — agent config name under configs/agents/ (e.g. "story_generator")
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunnerOutput:
    """Result of one case execution, before MinIO upload."""

    artifact_path: Optional[str] = None  # local file (image/video/audio)
    output_json: Optional[Any] = None    # structured output (llm_agent)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Image ─────────────────────────────────────────────────────────────────────


async def run_image_case(
    case_params: dict, model_id: str, model_config: dict, output_dir: str
) -> RunnerOutput:
    """case_params: Txt2ImageParams fields (prompt required)."""
    from virtual_streamer.image_generation.stable_cpp_client import (
        StableDiffusionCppClient,
        StableDiffusionCppConfig,
        Txt2ImageParams,
    )

    merged = {**model_config.get("params", {}), **case_params}
    if model_id == "openai":
        from virtual_streamer.image_generation.openai_image_client import (
            OpenAIImageClient,
            OpenAIImageConfig,
            OpenAITxt2ImageParams,
        )
        client = OpenAIImageClient(OpenAIImageConfig())
        allowed = OpenAITxt2ImageParams.model_fields.keys()
        params = OpenAITxt2ImageParams(**{k: v for k, v in merged.items() if k in allowed})
    else:
        config = StableDiffusionCppConfig(
            server_url=model_config.get(
                "server_url", os.environ.get("SD_SERVER_URL", "http://gx10-cbc5:1234")
            )
        )
        client = StableDiffusionCppClient(config)
        params = Txt2ImageParams(**merged)
    async with client:
        result = await client.txt2image(params, output_dir=output_dir)
    return RunnerOutput(
        artifact_path=result.image_path,
        metadata={"width": result.width, "height": result.height, "seed": result.seed},
    )


# ── Video ─────────────────────────────────────────────────────────────────────


async def run_video_case(
    case_params: dict, model_id: str, model_config: dict, output_dir: str
) -> RunnerOutput:
    """case_params: VideoGenerationParams fields (prompt required); model_id is a preset."""
    from virtual_streamer.video_generation.ltx_client import (
        LTXVideoConfig,
        VideoGenerationParams,
        WanGPLTXClient,
    )

    config = LTXVideoConfig(
        server_url=model_config.get(
            "server_url", os.environ.get("LTX_SERVER_URL", "http://gx10-cbc5:8082")
        ),
        timeout=float(model_config.get("timeout", os.environ.get("LTX_TIMEOUT", "3600.0"))),
    )
    preset = model_id if model_id in ("fast", "quality", "high_quality") else "fast"
    params = VideoGenerationParams.from_preset(
        preset, **{**model_config.get("params", {}), **case_params}
    )
    async with WanGPLTXClient(config) as client:
        result = await client.generate_video(params, output_dir=output_dir)
    return RunnerOutput(
        artifact_path=result.video_path,
        metadata={
            "duration_seconds": result.duration_seconds,
            "width": result.width,
            "height": result.height,
            "fps": result.fps,
        },
    )


# ── TTS ───────────────────────────────────────────────────────────────────────


async def run_tts_case(
    case_params: dict, model_id: str, model_config: dict, output_dir: str
) -> RunnerOutput:
    """case_params: {text, ...fish params}; model_id is a character_id ('default' = no cloning)."""
    from virtual_streamer.utils.utils import txt_to_speech_call_fish

    text = case_params["text"]
    tts_params = {k: v for k, v in case_params.items() if k != "text"}
    tts_params.update(model_config.get("params", {}))

    if model_id and model_id != "default":
        from virtual_streamer.utils.character_loader import load_character
        from virtual_streamer.api.dependencies import get_storage_resolver

        character = await load_character(model_id)
        if character.voice_samples:
            sample = character.voice_samples[0]
            tts_params["reference_audio"] = await get_storage_resolver().resolve_file(
                sample.sample_storage_path
            )
            tts_params["reference_text"] = sample.transcript

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{uuid.uuid4().hex}.wav")
    await asyncio.to_thread(
        txt_to_speech_call_fish, speech_lines=text, outpath=outpath, format="wav",
        **tts_params,
    )
    if not os.path.exists(outpath):
        raise RuntimeError("TTS call produced no audio file")
    return RunnerOutput(artifact_path=outpath)


# ── LLM agent ─────────────────────────────────────────────────────────────────


async def run_llm_agent_case(
    case_params: dict, model_id: str, model_config: dict, output_dir: str
) -> RunnerOutput:
    """
    case_params: {message: str, state: dict?}; model_id names configs/agents/{model_id}.yaml.

    Runs the agent through an in-memory ADK session (same flow as run_video_judge)
    and captures the final response text, JSON-parsed when possible.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from virtual_streamer.lib.agents import BaseLlmAgent

    agent = BaseLlmAgent(
        name=model_id,
        instruction=model_config.get("instruction") or case_params.get("instruction"),
    )
    session_service = InMemorySessionService()
    user_id = f"eval_{uuid.uuid4().hex[:8]}"
    session = await session_service.create_session(
        app_name="eval_bench",
        user_id=user_id,
        session_id=f"run_{uuid.uuid4().hex[:8]}",
        state=case_params.get("state") or {},
    )
    runner = Runner(agent=agent, app_name="eval_bench", session_service=session_service)
    content = types.Content(
        role="user", parts=[types.Part(text=case_params.get("message", ""))]
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        if event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final_text = "\n".join(texts)

    try:
        output = json.loads(final_text)
    except (json.JSONDecodeError, TypeError):
        output = {"text": final_text}
    return RunnerOutput(output_json=output)


# ── Optional judge (video/image benches with judge_agent set) ─────────────────


async def judge_artifact(
    judge_agent: str, artifact_path: str, case_params: dict
) -> Optional[dict]:
    """Run the named judge agent on an artifact. Only 'video_judge' is supported today."""
    if judge_agent != "video_judge":
        logger.warning(f"Unknown judge agent '{judge_agent}' — skipping auto-scoring")
        return None
    from virtual_streamer.agents.video_judge.agent import run_video_judge

    description = case_params.get("prompt") or json.dumps(case_params)
    verdict = await run_video_judge(artifact_path, description)
    return verdict.model_dump()


RUNNERS: Dict[str, Callable] = {
    "image": run_image_case,
    "video": run_video_case,
    "tts": run_tts_case,
    "llm_agent": run_llm_agent_case,
}
