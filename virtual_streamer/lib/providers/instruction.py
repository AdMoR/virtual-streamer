"""
Instruction provider base classes for ADK agents.

This module provides the InstructionProvider interface for generating
dynamic prompts based on runtime context. Use instruction providers when
you need prompts that vary based on:
- User session state
- Language/locale settings
- Feature flags
- Retrieved context
- Any other runtime data

Example:
    class LocalizedInstructionProvider(InstructionProvider):
        async def __call__(self, context: ReadonlyContext) -> str:
            language = context.state.get("language", "en")
            if language == "fr":
                return "Vous êtes un assistant serviable."
            return "You are a helpful assistant."

    @lru_cache
    def get_instruction_provider():
        return LocalizedInstructionProvider()
"""

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from string import Template
from typing import Any, Dict, Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.utils import instructions_utils

logger = logging.getLogger(__name__)


class InstructionProvider(ABC):
    """Abstract base class for dynamic instruction generation.

    InstructionProvider generates agent prompts at runtime based on the
    current context. This enables:
    - Language localization
    - User-specific customization
    - Feature flag-based behavior
    - Context injection from state

    The __call__ method is invoked by ADK before each agent execution,
    allowing the prompt to adapt to the current situation.

    Important: Use @lru_cache on factory functions to ensure singleton
    instances of your providers.

    Example:
        class MyInstructionProvider(InstructionProvider):
            async def __call__(self, context: ReadonlyContext) -> str:
                user_name = context.state.get("user_name", "User")
                return f"Help {user_name} with their questions."

        @lru_cache
        def get_my_provider():
            return MyInstructionProvider()
    """

    @abstractmethod
    async def __call__(self, context: ReadonlyContext) -> str:
        """Generate the instruction based on the current context.

        Args:
            context: ReadonlyContext providing access to session state.
                     Access state via context.state (dict-like).

        Returns:
            The instruction string to use for this agent execution.
        """
        ...


class StaticInstructionProvider(InstructionProvider):
    """Simple provider that returns a static instruction.

    Useful when you want to use the InstructionProvider interface
    but don't need dynamic behavior.

    Example:
        provider = StaticInstructionProvider(
            "You are a helpful assistant."
        )
    """

    def __init__(self, instruction: str):
        """Initialize with a static instruction.

        Args:
            instruction: The instruction to return.
        """
        self.instruction = instruction

    async def __call__(self, context: ReadonlyContext) -> str:
        return self.instruction


class TemplateInstructionProvider(InstructionProvider):
    """Provider that interpolates state variables into a template.

    Uses Python's string.Template for safe substitution of state
    values into the instruction template.

    Template syntax uses $variable_name or ${variable_name}.

    Example:
        provider = TemplateInstructionProvider(
            template="You are helping $user_name with $topic.",
            defaults={"user_name": "the user", "topic": "their query"},
        )
    """

    def __init__(
        self,
        template: str,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with a template and optional defaults.

        Args:
            template: Template string with $variable placeholders.
            defaults: Default values for template variables.
        """
        self.template = Template(template)
        self.defaults = defaults or {}

    async def __call__(self, context: ReadonlyContext) -> str:
        # Merge defaults with state (state takes precedence)
        values = {**self.defaults}

        # Extract state values
        for key in self.defaults.keys():
            if key in context.state:
                values[key] = context.state[key]

        # Safe substitute handles missing keys gracefully
        return self.template.safe_substitute(values)


class SessionStateInstructionProvider(InstructionProvider):
    """Provider that uses ADK's built-in state injection.

    This provider leverages google.adk.utils.instructions_utils
    to inject session state variables into the instruction template.

    Template syntax: {variable_name} for state injection.

    Example:
        provider = SessionStateInstructionProvider(
            template="Current user: {user_name}. Context: {context}"
        )
    """

    def __init__(self, template: str):
        """Initialize with a template.

        Args:
            template: Template with {variable} placeholders for state injection.
        """
        self.template = template

    async def __call__(self, context: ReadonlyContext) -> str:
        return await instructions_utils.inject_session_state(
            self.template, context
        )


class CompositeInstructionProvider(InstructionProvider):
    """Provider that combines multiple instruction sections.

    Useful for building complex prompts from modular components.
    Each section can be a string or another InstructionProvider.

    Example:
        provider = CompositeInstructionProvider(
            sections=[
                "# Role\nYou are a helpful assistant.",
                ContextProvider(),  # Dynamic context
                "# Output Format\nRespond in JSON.",
            ],
            separator="\\n\\n",
        )
    """

    def __init__(
        self,
        sections: list,
        separator: str = "\n\n",
    ):
        """Initialize with instruction sections.

        Args:
            sections: List of strings or InstructionProviders.
            separator: String to join sections with.
        """
        self.sections = sections
        self.separator = separator

    async def __call__(self, context: ReadonlyContext) -> str:
        parts = []

        for section in self.sections:
            if isinstance(section, str):
                parts.append(section)
            elif isinstance(section, InstructionProvider):
                parts.append(await section(context))
            elif callable(section):
                result = section(context)
                # Handle async callables
                if hasattr(result, "__await__"):
                    result = await result
                parts.append(str(result))
            else:
                parts.append(str(section))

        return self.separator.join(parts)


class ConditionalInstructionProvider(InstructionProvider):
    """Provider that selects instructions based on state conditions.

    Useful for feature flags, A/B testing, or user-specific behavior.

    Example:
        provider = ConditionalInstructionProvider(
            condition_key="user_tier",
            instructions={
                "premium": "You have access to advanced features...",
                "basic": "You are a basic assistant...",
            },
            default="You are a helpful assistant.",
        )
    """

    def __init__(
        self,
        condition_key: str,
        instructions: Dict[str, str],
        default: str = "",
    ):
        """Initialize with condition mapping.

        Args:
            condition_key: State key to check for condition value.
            instructions: Mapping of condition values to instructions.
            default: Default instruction if condition not matched.
        """
        self.condition_key = condition_key
        self.instructions = instructions
        self.default = default

    async def __call__(self, context: ReadonlyContext) -> str:
        condition_value = context.state.get(self.condition_key)

        if condition_value is not None:
            str_value = str(condition_value)
            if str_value in self.instructions:
                return self.instructions[str_value]

        return self.default


class FileInstructionProvider(InstructionProvider):
    """Provider that loads instructions from a file.

    Supports both static files and templates with state injection.
    Useful for managing long prompts outside of Python code.

    Example:
        provider = FileInstructionProvider(
            file_path="prompts/qa_agent.txt",
            use_template=True,
        )
    """

    def __init__(
        self,
        file_path: str,
        use_template: bool = False,
        encoding: str = "utf-8",
    ):
        """Initialize with file path.

        Args:
            file_path: Path to the instruction file.
            use_template: If True, apply state variable injection.
            encoding: File encoding.
        """
        self.file_path = file_path
        self.use_template = use_template
        self.encoding = encoding
        self._cached_content: Optional[str] = None

    def _load_file(self) -> str:
        """Load and cache file content."""
        if self._cached_content is None:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                self._cached_content = f.read()
        return self._cached_content

    async def __call__(self, context: ReadonlyContext) -> str:
        content = self._load_file()

        if self.use_template:
            return await instructions_utils.inject_session_state(content, context)

        return content

    def reload(self) -> None:
        """Force reload of the file on next call."""
        self._cached_content = None

