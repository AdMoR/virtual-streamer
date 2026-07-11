"""
Schemas for the VideoJudgeAgent.

JudgeVerdict is the structured output stored per candidate segment and
persisted in the segment_candidates table (judge_verdict JSON column).
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# State keys (scoped to the judge pipeline)
JUDGE_VIDEO_PATH = "judge_video_path"
JUDGE_SCENE_DESCRIPTION = "judge_scene_description"
JUDGE_VERDICT = "judge_verdict"

ArtifactCategory = Literal[
    "unrealistic_movement",   # e.g. person running backward, limbs bending wrong way
    "impossible_body",        # extra/missing limbs, merged faces, deformed anatomy
    "object_intersection",    # body passing through objects, clipping
    "impossible_setting",     # physically impossible scene layout
    "identity_drift",         # character changes appearance mid-clip
    "visual_glitch",          # flicker, smearing, heavy distortion
    "other",
]


class JudgeArtifact(BaseModel):
    """One detected generation artifact."""

    category: ArtifactCategory
    description: str = Field(description="What is wrong, in one sentence")
    severity: Literal["minor", "major", "blocking"] = Field(
        description="blocking = video unusable, major = clearly visible, minor = subtle"
    )


class JudgeVerdict(BaseModel):
    """Structured judgement of one generated video segment."""

    passed: bool = Field(description="True if the segment is usable in the final video")
    score: float = Field(ge=0.0, le=10.0, description="Overall quality 0-10")
    artifacts: List[JudgeArtifact] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of the verdict")
    judge_error: Optional[str] = Field(
        default=None,
        description="Set when the judge itself failed (verdict is then a permissive default)",
    )

    @classmethod
    def permissive_default(cls, error: str) -> "JudgeVerdict":
        """Fallback verdict when the judge fails — never blocks generation."""
        return cls(
            passed=True,
            score=5.0,
            artifacts=[],
            reasoning="Judge unavailable — segment accepted by default.",
            judge_error=error,
        )
