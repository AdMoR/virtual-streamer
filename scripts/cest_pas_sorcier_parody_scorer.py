import os
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import anthropic
import mlflow

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "cest_pas_sorcier_parody_evaluation"
mlflow.set_experiment(EXPERIMENT_NAME)


@dataclass
class RubricScore:
    """Represents a score for a single rubric dimension"""

    dimension: str
    score: int
    reasoning: str
    max_score: int = 5


@dataclass
class RubricResult:
    """Represents the complete scoring result for a rubric"""

    rubric_name: str
    scores: List[RubricScore]
    total_score: int
    max_total_score: int
    feedback: str


@dataclass
class ParodyEvaluation:
    """Complete evaluation result for a parody story"""

    story: str
    rubric_1_result: RubricResult
    rubric_2_result: RubricResult
    rubric_3_result: RubricResult
    overall_score: float
    overall_feedback: str


class CestPasSorcierParodyScorer:
    """
    Scores parody stories based on three rubrics:
    1. Character Voice & Personality Authenticity
    2. Comedic Structure & Absurdity
    3. Cultural & Contextual Relevance

    Prompts are stored and retrieved from MLflow for version control.
    """

    RUBRIC_1_PROMPT_NAME = "rubric_1_character_voice"
    RUBRIC_2_PROMPT_NAME = "rubric_2_comedic_structure"
    RUBRIC_3_PROMPT_NAME = "rubric_3_cultural_relevance"

    RUBRIC_1_PROMPT_TEMPLATE = """You are an expert evaluator of "C'est pas Sorcier" parody humor.

Evaluate the following story based on RUBRIC 1: Character Voice & Personality Authenticity

RUBRIC 1 Dimensions:
1. Fred's Bombastic Energy (1-5): Does Fred display overconfident, grandiose schemes? Does he reference past glories (C'est pas Sorcier era)? Does he use casual French slang ("flouze," "pécho," "kiffer")?
2. Jamy's Passive Role (1-5): Is Jamy positioned as the skeptical listener/straight man? Does he receive Fred's ideas without much pushback?
3. Nostalgic References (1-5): Are there callbacks to 90s France 3, specific locations, or the show's heyday? Do they feel organic to the bit?
4. Tone Consistency (1-5): Is the humor maintained through absurdity rather than meanness? Does it feel affectionate toward the characters?

STORY TO EVALUATE:
{story}

Provide your evaluation in the following JSON format:
{{
    "scores": [
        {{"dimension": "Fred's Bombastic Energy", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "Jamy's Passive Role", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "Nostalgic References", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "Tone Consistency", "score": <1-5>, "reasoning": "<explanation>"}}
    ],
    "feedback": "<overall feedback for this rubric>"
}}"""

    RUBRIC_2_PROMPT_TEMPLATE = """You are an expert evaluator of "C'est pas Sorcier" parody humor.

Evaluate the following story based on RUBRIC 2: Comedic Structure & Absurdity

RUBRIC 2 Dimensions:
1. Escalation Logic (1-5): Does the idea start somewhat plausible then spiral into ridiculous territory?
2. Specific Details (1-5): Are there concrete, oddly specific elements? (Nokia 3310, Bourg-en-Gonesse, 1998 New Year's party, TGV reference)
3. Pseudo-Scientific Justification (1-5): Does Fred justify absurd ideas with faux-logic or "scientific" reasoning from the show?
4. Self-Aware Humor (1-5): Does the parody acknowledge the ridiculousness without breaking character?

STORY TO EVALUATE:
{story}

Provide your evaluation in the following JSON format:
{{
    "scores": [
        {{"dimension": "Escalation Logic", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "Specific Details", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "Pseudo-Scientific Justification", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "Self-Aware Humor", "score": <1-5>, "reasoning": "<explanation>"}}
    ],
    "feedback": "<overall feedback for this rubric>"
}}"""

    RUBRIC_3_PROMPT_TEMPLATE = """You are an expert evaluator of "C'est pas Sorcier" parody humor.

Evaluate the following story based on RUBRIC 3: Cultural & Contextual Relevance

RUBRIC 3 Dimensions:
1. Modern vs. Retro Clash (1-5): Does it juxtapose outdated Fred/Jamy energy with current trends? (TikTok, Red Bull, energy drinks, "bobos")
2. French Cultural Specificity (1-5): Does it use French slang, locations, and cultural references authentically?
3. Grandiose Delusion (1-5): Does Fred pitch a half-baked idea with unwarranted confidence? Will this idea bring some kind of fame to Fred ? 
4. Accessibility (1-5): Would someone unfamiliar with C'est pas Sorcier still find it funny, or is it primarily for nostalgic fans?

STORY TO EVALUATE:
{story}

Provide your evaluation in the following JSON format:
{{
    "scores": [
        {{"dimension": "Modern vs. Retro Clash", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "French Cultural Specificity", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "Entrepreneurial Delusion", "score": <1-5>, "reasoning": "<explanation>"}},
        {{"dimension": "Accessibility", "score": <1-5>, "reasoning": "<explanation>"}}
    ],
    "feedback": "<overall feedback for this rubric>"
}}"""

    def __init__(self):
        """Initialize the scorer"""
        self.model = "claude-3-5-sonnet-20241022"
        self._initialize_prompts_in_mlflow()

    def _initialize_prompts_in_mlflow(self):
        """Initialize prompts in MLflow if they don't exist"""
        try:
            # Check if prompts already exist in MLflow
            self._get_prompt_from_mlflow(self.RUBRIC_1_PROMPT_NAME)
        except Exception:
            # If not, register them
            self._register_prompt_in_mlflow(
                self.RUBRIC_1_PROMPT_NAME,
                self.RUBRIC_1_PROMPT_TEMPLATE,
                "Rubric 1: Character Voice & Personality Authenticity",
            )
            self._register_prompt_in_mlflow(
                self.RUBRIC_2_PROMPT_NAME,
                self.RUBRIC_2_PROMPT_TEMPLATE,
                "Rubric 2: Comedic Structure & Absurdity",
            )
            self._register_prompt_in_mlflow(
                self.RUBRIC_3_PROMPT_NAME,
                self.RUBRIC_3_PROMPT_TEMPLATE,
                "Rubric 3: Cultural & Contextual Relevance",
            )

    def _register_prompt_in_mlflow(
        self, prompt_name: str, prompt_template: str, description: str
    ):
        """Register a prompt in MLflow"""
        with mlflow.start_run():
            mlflow.log_param("prompt_name", prompt_name)
            mlflow.log_param("description", description)
            mlflow.log_text(prompt_template, f"{prompt_name}.txt")
            mlflow.set_tag("prompt_type", "evaluation_rubric")

    def _get_prompt_from_mlflow(self, prompt_name: str) -> str:
        """Retrieve a prompt from MLflow"""
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            raise ValueError(f"Experiment {EXPERIMENT_NAME} not found")

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.prompt_name = '{prompt_name}'",
            max_results=1,
        )

        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            artifacts = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="", dst_path=None
            )
            # Return the template based on prompt name
            if prompt_name == self.RUBRIC_1_PROMPT_NAME:
                return self.RUBRIC_1_PROMPT_TEMPLATE
            elif prompt_name == self.RUBRIC_2_PROMPT_NAME:
                return self.RUBRIC_2_PROMPT_TEMPLATE
            elif prompt_name == self.RUBRIC_3_PROMPT_NAME:
                return self.RUBRIC_3_PROMPT_TEMPLATE

        # Fallback to template if not found in MLflow
        if prompt_name == self.RUBRIC_1_PROMPT_NAME:
            return self.RUBRIC_1_PROMPT_TEMPLATE
        elif prompt_name == self.RUBRIC_2_PROMPT_NAME:
            return self.RUBRIC_2_PROMPT_TEMPLATE
        elif prompt_name == self.RUBRIC_3_PROMPT_NAME:
            return self.RUBRIC_3_PROMPT_TEMPLATE

        raise ValueError(f"Prompt {prompt_name} not found")

    def _call_claude_for_rubric(self, prompt: str) -> Dict:
        """Call Claude API to evaluate a rubric"""
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON from response
        response_text = message.content[0].text

        # Try to parse JSON from the response
        try:
            # Find JSON in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        return {"scores": [], "feedback": "Error parsing response"}

    def _parse_rubric_response(self, response: Dict, rubric_name: str) -> RubricResult:
        """Parse Claude's response into a RubricResult"""
        scores = []
        total_score = 0

        for score_data in response.get("scores", []):
            score = RubricScore(
                dimension=score_data.get("dimension", "Unknown"),
                score=score_data.get("score", 0),
                reasoning=score_data.get("reasoning", ""),
                max_score=5,
            )
            scores.append(score)
            total_score += score.score

        max_total = len(scores) * 5

        return RubricResult(
            rubric_name=rubric_name,
            scores=scores,
            total_score=total_score,
            max_total_score=max_total,
            feedback=response.get("feedback", ""),
        )

    def evaluate_story(self, story: str) -> ParodyEvaluation:
        """
        Evaluate a parody story using all three rubrics

        Args:
            story: The parody story text to evaluate

        Returns:
            ParodyEvaluation object with all scores and feedback
        """
        with mlflow.start_run():
            # Log the story being evaluated
            mlflow.log_text(story, "story_evaluated.txt")

            print(
                "Evaluating story with Rubric 1: Character Voice & Personality Authenticity..."
            )
            rubric_1_prompt = self._get_prompt_from_mlflow(self.RUBRIC_1_PROMPT_NAME)
            rubric_1_response = self._call_claude_for_rubric(
                rubric_1_prompt.format(story=story)
            )
            rubric_1_result = self._parse_rubric_response(
                rubric_1_response, "Character Voice & Personality Authenticity"
            )

            print("Evaluating story with Rubric 2: Comedic Structure & Absurdity...")
            rubric_2_prompt = self._get_prompt_from_mlflow(self.RUBRIC_2_PROMPT_NAME)
            rubric_2_response = self._call_claude_for_rubric(
                rubric_2_prompt.format(story=story)
            )
            rubric_2_result = self._parse_rubric_response(
                rubric_2_response, "Comedic Structure & Absurdity"
            )

            print("Evaluating story with Rubric 3: Cultural & Contextual Relevance...")
            rubric_3_prompt = self._get_prompt_from_mlflow(self.RUBRIC_3_PROMPT_NAME)
            rubric_3_response = self._call_claude_for_rubric(
                rubric_3_prompt.format(story=story)
            )
            rubric_3_result = self._parse_rubric_response(
                rubric_3_response, "Cultural & Contextual Relevance"
            )

            # Calculate overall score
            total_score = (
                rubric_1_result.total_score
                + rubric_2_result.total_score
                + rubric_3_result.total_score
            )
            max_total = (
                rubric_1_result.max_total_score
                + rubric_2_result.max_total_score
                + rubric_3_result.max_total_score
            )
            overall_score = (total_score / max_total) * 100 if max_total > 0 else 0

            # Generate overall feedback
            overall_feedback = self._generate_overall_feedback(
                rubric_1_result, rubric_2_result, rubric_3_result, overall_score
            )

            # Log metrics to MLflow
            mlflow.log_metric("overall_score", overall_score)
            mlflow.log_metric("rubric_1_score", rubric_1_result.total_score)
            mlflow.log_metric("rubric_2_score", rubric_2_result.total_score)
            mlflow.log_metric("rubric_3_score", rubric_3_result.total_score)

            # Log individual dimension scores
            for score in rubric_1_result.scores:
                mlflow.log_metric(
                    f"rubric_1_{score.dimension.lower().replace(' ', '_')}", score.score
                )
            for score in rubric_2_result.scores:
                mlflow.log_metric(
                    f"rubric_2_{score.dimension.lower().replace(' ', '_')}", score.score
                )
            for score in rubric_3_result.scores:
                mlflow.log_metric(
                    f"rubric_3_{score.dimension.lower().replace(' ', '_')}", score.score
                )

            return ParodyEvaluation(
                story=story,
                rubric_1_result=rubric_1_result,
                rubric_2_result=rubric_2_result,
                rubric_3_result=rubric_3_result,
                overall_score=overall_score,
                overall_feedback=overall_feedback,
            )

    def _generate_overall_feedback(
        self, r1: RubricResult, r2: RubricResult, r3: RubricResult, score: float
    ) -> str:
        """Generate overall feedback based on all rubric results"""
        if score >= 80:
            quality = "Excellent"
        elif score >= 65:
            quality = "Good"
        elif score >= 50:
            quality = "Fair"
        else:
            quality = "Needs Improvement"

        return (
            f"{quality} parody with {score:.1f}% alignment to C'est pas Sorcier spirit."
        )

    def format_evaluation_report(self, evaluation: ParodyEvaluation) -> str:
        """Format the evaluation into a readable report"""
        report = []
        report.append("=" * 80)
        report.append("C'EST PAS SORCIER PARODY EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall Score
        report.append(f"OVERALL SCORE: {evaluation.overall_score:.1f}%")
        report.append(f"OVERALL FEEDBACK: {evaluation.overall_feedback}")
        report.append("")

        # Rubric 1
        report.append("-" * 80)
        report.append(f"RUBRIC 1: {evaluation.rubric_1_result.rubric_name}")
        report.append(
            f"Score: {evaluation.rubric_1_result.total_score}/{evaluation.rubric_1_result.max_total_score}"
        )
        report.append("")
        for score in evaluation.rubric_1_result.scores:
            report.append(f"  • {score.dimension}: {score.score}/5")
            report.append(f"    {score.reasoning}")
        report.append(f"\nFeedback: {evaluation.rubric_1_result.feedback}")
        report.append("")

        # Rubric 2
        report.append("-" * 80)
        report.append(f"RUBRIC 2: {evaluation.rubric_2_result.rubric_name}")
        report.append(
            f"Score: {evaluation.rubric_2_result.total_score}/{evaluation.rubric_2_result.max_total_score}"
        )
        report.append("")
        for score in evaluation.rubric_2_result.scores:
            report.append(f"  • {score.dimension}: {score.score}/5")
            report.append(f"    {score.reasoning}")
        report.append(f"\nFeedback: {evaluation.rubric_2_result.feedback}")
        report.append("")

        # Rubric 3
        report.append("-" * 80)
        report.append(f"RUBRIC 3: {evaluation.rubric_3_result.rubric_name}")
        report.append(
            f"Score: {evaluation.rubric_3_result.total_score}/{evaluation.rubric_3_result.max_total_score}"
        )
        report.append("")
        for score in evaluation.rubric_3_result.scores:
            report.append(f"  • {score.dimension}: {score.score}/5")
            report.append(f"    {score.reasoning}")
        report.append(f"\nFeedback: {evaluation.rubric_3_result.feedback}")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Main function to demonstrate the scorer"""

    # Example stories
    example_story_1 = """Eh bien, dis donc Jamy, aujourd'hui, j'ai une idée qui va te retourner ton caleçon ! Tu sais quoi ? Je veux lancer ma propre chaîne de twerking sur TikTok ! Eh oui, twerking, le truc où on se frotte le cul sur quelqu'un d'autre, tu sais, comme à nos soirées bien arrosées avec Marcel.

Hier soir, je me suis enfilé cinq heures de TikTok, et je te jure, Jamy, c'est un véritable festival du boule ! Tu peux en croire mon instinct du bizness, Jamy, on va se faire du flouze comme à la grande époque de c'est pas sorcier !

Pour commencer, j'ai mis les petits plats dans les grands et j'ai loué la salle des fêtes de Bourg-en-Gonesse. Tu sais Jamy, la grande salle où on a fait la fête pour le Nouvel An de 1998 avec la rédaction de France 3 Poitou charente. Et pour filmer tout ça, j'ai emprunté le Nokia trente trois dix de ma cousine. C'est pas le dernier cri, mais ça fait le job !

Et tu sais quoi, Jamy ? Avec notre école de twerk, tu vas enfin pouvoir pécho ! Oui, parce que le twerk, ça attire les gonzesses, c'est un fait. Les filles adorent ça, et si tu leur montres tes maquettes, je suis sur qu'une ou deux se laisseront bien ramener au camion.

Alors, qu'est-ce que t'en penses, Jamy ?"""

    scorer = CestPasSorcierParodyScorer()

    print("Starting evaluation...")
    print("")

    evaluation = scorer.evaluate_story(example_story_1)
    report = scorer.format_evaluation_report(evaluation)
    print(report)

    # Save report to file
    output_file = "parody_evaluation_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {output_file}")
    print(f"\nMLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print("View results at: http://localhost:5000")


if __name__ == "__main__":
    main()
