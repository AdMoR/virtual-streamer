from pydantic import BaseModel, Field
from typing import Optional


class Subject(BaseModel):
    description: str = Field(
        description="Detailed visual description of the subject including material, color, and distinguishing features.",
        examples=["Minimalist ceramic coffee mug with bright red steam rising from hot coffee inside"],
    )
    pose: Optional[str] = Field(
        default=None,
        description="Pose or stance of the subject.",
        examples=["Stationary on surface"],
    )
    position: str = Field(
        description="Where the subject is placed within the frame.",
        examples=["Center foreground on polished concrete surface"],
    )
    color_palette: Optional[list[str]] = Field(
        default=None,
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
        default=None,
        description="What is in sharp focus and any notable focus behavior.",
        examples=["Sharp focus on steam rising from coffee and both mugs in frame"],
    )
    lens_mm: Optional[int] = Field(
        default=None,
        alias="lens-mm",
        description="Focal length of the lens in millimeters.",
        examples=[85, 50, 35],
    )
    f_number: Optional[str] = Field(
        default=None,
        alias="f-number",
        description="Aperture f-stop value.",
        examples=["f/5.6", "f/2.8", "f/11"],
    )
    ISO: Optional[int] = Field(
        default=None,
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
        default=None,
        description="Artistic or photographic style directive.",
        examples=["Ultra-realistic product photography with commercial quality"],
    )
    color_palette: Optional[list[str]] = Field(
        default=None,
        description="Global color palette for the entire scene.",
        examples=[["matte black", "matte yellow", "soft white highlights"]],
    )
    lighting: str = Field(
        description="Lighting setup and quality.",
        examples=["Three-point softbox setup creating soft, diffused highlights with no harsh shadows"],
    )
    mood: Optional[str] = Field(
        default=None,
        description="Emotional tone or atmosphere of the image.",
        examples=["Clean, professional, minimalist"],
    )
    background: Optional[str] = Field(
        default=None,
        description="Background environment and surface details.",
        examples=["Polished concrete surface with studio backdrop"],
    )
    composition: Optional[str] = Field(
        default=None,
        description="Compositional technique or framing rule.",
        examples=["rule of thirds", "centered symmetry", "leading lines"],
    )
    camera: Camera = Field(
        description="Camera and lens settings for the shot.",
    )

    def to_prompt(self) -> str:
        """Convert to a flat string prompt suitable for txt2video / image generation."""
        parts = []

        # Camera framing first — strongly influences composition
        camera_parts = [self.camera.angle, self.camera.distance]
        if self.camera.focus:
            camera_parts.append(self.camera.focus)
        if self.camera.lens_mm is not None:
            camera_parts.append(f"{self.camera.lens_mm}mm lens")
        if self.camera.f_number:
            camera_parts.append(self.camera.f_number)
        if self.camera.ISO is not None:
            camera_parts.append(f"ISO {self.camera.ISO}")
        parts.append(", ".join(camera_parts))

        parts.append(self.scene)

        for subject in self.subjects:
            subject_parts = [subject.description, f"at {subject.position}"]
            if subject.pose:
                subject_parts.append(subject.pose)
            if subject.color_palette:
                subject_parts.append(f"colors: {', '.join(subject.color_palette)}")
            parts.append(", ".join(subject_parts))

        parts.append(self.lighting)

        if self.background:
            parts.append(self.background)

        if self.composition:
            parts.append(f"Composition: {self.composition}")

        if self.style:
            parts.append(self.style)

        if self.mood:
            parts.append(self.mood)

        if self.color_palette:
            parts.append(f"Color palette: {', '.join(self.color_palette)}")

        return ". ".join(parts) + "."
