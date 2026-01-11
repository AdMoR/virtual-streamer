from pydantic import BaseModel, Field

class RubricExample(BaseModel):
    """A self-contained example illustrating a rubric grading."""
    example: str = Field(
        description="An example used to define if the Rubrics will pass or not"
    )
    grade: bool = Field(
        description="Is the example passing or not the criteria"
    )
    grading_explanation: str = Field(
        description="Explanation of the grading of the example"
    )


class Rubric(BaseModel):
    """A  rubric that applies to a news articles."""
    description: str = Field(
        description="Description détaillée et actionnable du critère. "
                    "Un relecteur doit pouvoir évaluer un article en lisant uniquement cette description."
    )
    examples: list[RubricExample] = Field(
        description="Meilleurs exemples fusionnés des différentes catégories, "
                    "autonomes et compréhensibles sans contexte original.",
        default_factory=list,
    )


class MapPhaseOutput(BaseModel):
    """Output schema for the map phase (per-category rubric extraction)."""
    rubrics: list[Rubric] = Field(
        description="Liste des critères de qualité identifiés pour cette catégorie"
    )


class ReducePhaseOutput(BaseModel):
    """Output schema for the reduce phase (consolidated rubrics)."""

    general_rubrics: list[Rubric] = Field(
        description="Critères généraux applicables à tous les articles du Gorafi"
    )
    category_rubrics: dict[str, list[Rubric]] = Field(
        description="Critères spécifiques par catégorie",
        default_factory=dict,
    )
