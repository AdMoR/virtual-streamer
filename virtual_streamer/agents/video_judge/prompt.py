"""Prompt for the VideoJudgeAgent."""

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.agents.video_judge.schema import JUDGE_SCENE_DESCRIPTION

JUDGE_PROMPT_TEMPLATE = """You are a strict quality judge for AI-generated video segments.
You are shown N frames sampled evenly (in temporal order) from ONE short generated clip.

Intended scene description:
---
{scene_description}
---

Inspect the frames for generation artifacts. Focus on:
1. UNREALISTIC MOVEMENTS — inferred from frame-to-frame changes: a person running or
   walking backward when they should move forward, limbs bending impossibly,
   teleporting objects, motion inconsistent with the described action.
2. IMPOSSIBLE BODIES — extra or missing limbs/fingers, two heads, merged faces,
   deformed anatomy, a body partly inside furniture/walls/objects (clipping).
3. IMPOSSIBLE SETTINGS — objects floating without support, scene layout that could
   not physically exist, scale errors (giant hands, tiny doors).
4. IDENTITY DRIFT — the character's face/clothing changes noticeably between frames.
5. VISUAL GLITCHES — heavy smearing, flicker, duplicated textures, garbled text.

Also check the clip roughly matches the intended scene description.

Scoring guide:
- 9-10: clean, no visible artifacts, matches the description.
- 7-8: minor artifacts only; usable.
- 5-6: visible artifacts but the clip could pass with a tolerant viewer.
- 3-4: major artifacts (wrong motion direction, clipping, anatomy errors).
- 0-2: blocking artifacts; clip is unusable.

`passed` must be false when any artifact has severity "blocking", or when the
overall score is below 6.

Respond with ONLY a JSON object matching this schema (no markdown fences):
{{
  "passed": bool,
  "score": float,               // 0-10
  "artifacts": [
    {{"category": "unrealistic_movement|impossible_body|object_intersection|impossible_setting|identity_drift|visual_glitch|other",
      "description": "...",
      "severity": "minor|major|blocking"}}
  ],
  "reasoning": "one or two sentences"
}}
"""


class JudgeInstructionProvider:
    """Injects the scene description from state into the judge prompt."""

    def __call__(self, ctx: ReadonlyContext) -> str:
        scene_description = ctx.state.get(JUDGE_SCENE_DESCRIPTION) or "(none provided)"
        return JUDGE_PROMPT_TEMPLATE.format(scene_description=scene_description)
