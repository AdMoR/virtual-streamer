"""Schema for Location Builder Agent output."""

from pydantic import BaseModel, Field


class LocationDescriptionOutput(BaseModel):
    """Structured output from the location_formatter agent."""
    name: str
    description: str = Field(
        description=(
            "A complete diffusion-model prompt describing the location environment. "
            "Must cover: lighting, atmosphere, color palette, architectural/natural style, "
            "textures, foreground/background elements. "
            "Do NOT include any character or person descriptions. "
            "Example: 'A sunlit medieval castle courtyard with weathered stone walls, "
            "ivy-covered battlements, warm golden hour lighting, cobblestone floor, "
            "wooden market stalls visible in the background, cinematic wide angle.'"
        )
    )