from pydantic import BaseModel, Field
from typing import Optional


class Subject(BaseModel):
    description: str = Field(
        description="Detailed visual description of the subject including material, color, and distinguishing features.",
        examples=["Minimalist ceramic coffee mug with bright red steam rising from hot coffee inside"],
    )
    pose: Optional[str] = Field(
        description="Pose or stance of the subject.",
        examples=["Stationary on surface"],
    )
    position: str = Field(
        description="Where the subject is placed within the frame.",
        examples=["Center foreground on polished concrete surface"],
    )
    color_palette: Optional[list[str]] = Field(
        description="Dominant colors specific to this subject.",
        examples=[["matte black ceramic", "bright red steam"]],
    )


class Camera(BaseModel):
    angle: str = Field(
        description="Camera angle relative to the subject.",
        examples=["high angle", "eye level", "low angle"],
    )
    distance: str = Field(
        description="Shot distance / framing.",
        examples=["medium shot", "close-up", "wide shot"],
    )
    focus: Optional[str] = Field(
        description="What is in sharp focus and any notable focus behavior.",
        examples=["Sharp focus on steam rising from coffee and both mugs in frame"],
    )
    lens_mm: Optional[int] = Field(
        alias="lens-mm",
        description="Focal length of the lens in millimeters.",
        examples=[85, 50, 35],
    )
    f_number: Optional[str] = Field(
        alias="f-number",
        description="Aperture f-stop value.",
        examples=["f/5.6", "f/2.8", "f/11"],
    )
    ISO: Optional[int] = Field(
        description="Camera ISO sensitivity.",
        examples=[200, 400, 800],
    )

    model_config = {"populate_by_name": True}


class FluxPrompt(BaseModel):
    scene: str = Field(
        description="Overall scene context and environment.",
        examples=["Professional product photography setup with polished concrete surface"],
    )
    subjects: list[Subject] = Field(
        description="All subjects present in the image, ordered by visual prominence.",
    )
    style: Optional[str] = Field(
        description="Artistic or photographic style directive.",
        examples=["Ultra-realistic product photography with commercial quality"],
    )
    color_palette: Optional[list[str]] = Field(
        description="Global color palette for the entire scene.",
        examples=[["matte black", "matte yellow", "soft white highlights"]],
    )
    lighting: str = Field(
        description="Lighting setup and quality.",
        examples=["Three-point softbox setup creating soft, diffused highlights with no harsh shadows"],
    )
    mood: Optional[str] = Field(
        description="Emotional tone or atmosphere of the image.",
        examples=["Clean, professional, minimalist"],
    )
    background: Optional[str] = Field(
        description="Background environment and surface details.",
        examples=["Polished concrete surface with studio backdrop"],
    )
    composition: Optional[str] = Field(
        description="Compositional technique or framing rule.",
        examples=["rule of thirds", "centered symmetry", "leading lines"],
    )
    camera: Camera = Field(
        description="Camera and lens settings for the shot.",
    )
