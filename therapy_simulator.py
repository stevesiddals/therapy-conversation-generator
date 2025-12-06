# therapy_simulator_updated.py

import random
from openai import OpenAI
import google.generativeai as genai  # type: ignore
import anthropic
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict
from datetime import datetime


class ModelInterface(ABC):
    @abstractmethod
    def generate_message(self, system: str, messages: list, max_tokens: int, temperature: float) -> str:
        pass


class ClaudeModel(ModelInterface):
    """Base class for Claude models with optional extended thinking support."""

    def __init__(self, api_key: str, model_name: str, display_name: str, thinking_budget: Optional[int] = None):
        self.client = anthropic.Client(api_key=api_key)
        self.MODEL_NAME = model_name
        self.DISPLAY_NAME = display_name
        self.thinking_budget = thinking_budget

    def generate_message(self, system: str, messages: list, max_tokens: int, temperature: float) -> str:
        params = {
            "model": self.MODEL_NAME,
            "system": system,
            "messages": messages,
        }

        # Add thinking configuration if specified
        if self.thinking_budget is not None:
            # max_tokens must be greater than thinking budget
            # Add buffer to ensure max_tokens > thinking budget
            adjusted_max_tokens = max(max_tokens, self.thinking_budget + 100)
            params["max_tokens"] = adjusted_max_tokens
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
            # Temperature must be 1.0 when thinking is enabled
            params["temperature"] = 1.0
        else:
            params["max_tokens"] = max_tokens
            # Use provided temperature only when thinking is disabled
            params["temperature"] = temperature

        response = self.client.messages.create(**params)
        return self._extract_text_content(response)

    @staticmethod
    def _extract_text_content(response) -> str:
        text_content = []
        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
            # Skip thinking blocks - we only want the final response
        return " ".join(text_content).strip()


class GPTModel(ModelInterface):
    """Base class for GPT models with optional reasoning effort support."""

    def __init__(self, api_key: str, model_name: str, display_name: str, reasoning_effort: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.MODEL_NAME = model_name
        self.DISPLAY_NAME = display_name
        self.reasoning_effort = reasoning_effort

    def generate_message(self, system: str, messages: list, max_tokens: int, temperature: float) -> str:
        openai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

        params = {
            "model": self.MODEL_NAME,
            "messages": openai_messages,
            "temperature": temperature
        }

        # GPT-5.1 uses max_completion_tokens instead of max_tokens
        if self.MODEL_NAME.startswith("gpt-5"):
            # BUG FIX: Even with reasoning_effort='none', GPT-5.1 sometimes still uses reasoning tokens
            # This appears to be a known bug where reasoning_effort is not respected
            # Solution: Don't use reasoning_effort parameter at all, and provide 4x token budget
            # to ensure enough room for actual output after any hidden reasoning
            params["max_completion_tokens"] = max_tokens * 4
        else:
            params["max_tokens"] = max_tokens

        # NOTE: Removed reasoning_effort parameter entirely due to bugs where it's not respected

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class GeminiModel(ModelInterface):
    """Base class for Gemini models with optional thinking support."""

    def __init__(self, api_key: str, model_name: str, display_name: str, thinking_level: Optional[str] = None):
        genai.configure(api_key=api_key)
        self.MODEL_NAME = model_name
        self.DISPLAY_NAME = display_name
        self.thinking_level = thinking_level
        self.model = genai.GenerativeModel(self.MODEL_NAME)

    def generate_message(self, system: str, messages: list, max_tokens: int, temperature: float) -> str:
        combined_prompt = f"{system}\n\n"
        for msg in messages:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"

        response = self.model.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
        return response.text


class GeminiThinkingModel(ModelInterface):
    """Gemini model with thinking configuration using the new Python SDK."""

    def __init__(self, api_key: str, model_name: str, display_name: str, thinking_level: str = "low"):
        # Import the new Google GenAI SDK
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
        except:
            pass

        self.MODEL_NAME = model_name
        self.DISPLAY_NAME = display_name
        self.thinking_level = thinking_level
        self.api_key = api_key

    def generate_message(self, system: str, messages: list, max_tokens: int, temperature: float) -> str:
        # Use REST API directly for Gemini 3 with thinking_level
        import requests
        import json

        combined_prompt = f"{system}\n\n"
        for msg in messages:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [{"text": combined_prompt}]
            }],
            "generationConfig": {
                # BUG FIX: For Gemini 3 Pro, maxOutputTokens includes thinking tokens + output tokens
                # Gemini 3 with thinking_level can use significant amounts of thinking tokens
                # Increase by 3x to ensure enough room for actual output after thinking
                "maxOutputTokens": max_tokens * 3,
                "temperature": temperature
            }
        }

        # Add thinking config if thinking_level is specified
        if self.thinking_level:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": self.thinking_level.upper()
            }

        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

        if response.status_code != 200:
            raise ValueError(f"Gemini API error: {response.status_code} - {response.text}")

        result = response.json()

        # Handle cases where response might not have text
        if "candidates" not in result or not result["candidates"]:
            raise ValueError("No response candidates returned from Gemini")

        candidate = result["candidates"][0]

        if "content" not in candidate or "parts" not in candidate["content"]:
            # Check finish_reason
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if finish_reason == "MAX_TOKENS":
                raise ValueError(
                    "Response was cut off - token budget exhausted. The model ran out of tokens before completing the response.")
            elif finish_reason == "SAFETY":
                raise ValueError("Response blocked by safety filters")
            elif finish_reason == "RECITATION":
                raise ValueError("Response blocked due to recitation")
            else:
                raise ValueError(f"No text in response. Finish reason: {finish_reason}")

        # Extract text from parts
        text_parts = []
        for part in candidate["content"]["parts"]:
            if "text" in part:
                text_parts.append(part["text"])

        if not text_parts:
            raise ValueError("No text content in response")

        return " ".join(text_parts)


@dataclass
class PromptConfig:
    therapist_system: str = "You are a therapist."
    client_system: str = "You are a client in a therapy session."

    therapist_context_template: str = """You are practicing {approach}, with a {style} style. 
Remain compassionate and validating, providing a safe space for the client to explore their experiences. 
What follows is the therapy conversation so far."""

    client_context_template: str = """You are {name}, {age} years old and {gender}. 
You came to therapy because {presenting_problem}. 
Your context: {context}. What follows is the therapy conversation so far."""

    therapist_instruction: str = "Now respond as the therapist."
    client_instruction: str = "Take a moment to process what the therapist said, then respond naturally as yourself."


@dataclass
class ClientProfile:
    name: str
    age: int
    gender: str
    presenting_problem: str
    context: str

    def format_context(self, template: str) -> str:
        return template.format(
            name=self.name,
            age=self.age,
            gender=self.gender,
            presenting_problem=self.presenting_problem,
            context=self.context
        )


@dataclass
class TherapistProfile:
    approach: str
    style: str

    def format_context(self, template: str) -> str:
        return template.format(
            approach=self.approach,
            style=self.style
        )


@dataclass
class SessionMetadata:
    timestamp: str
    model: str
    max_tokens: int
    temperature: float
    prompt_config: PromptConfig
    client_profile: ClientProfile
    therapist_profile: TherapistProfile
    num_exchanges: int

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prompt_config": asdict(self.prompt_config),
            "client_profile": asdict(self.client_profile),
            "therapist_profile": asdict(self.therapist_profile),
            "num_exchanges": self.num_exchanges
        }


@dataclass
class TherapySession:
    metadata: SessionMetadata
    conversation: List[Dict[str, str]]


class TherapySessionGenerator:
    """
    Generates therapy conversation sessions using various AI models.

    Supported models:
    - OpenAI: GPT-4o, GPT-5.1
    - Anthropic: Claude Sonnet 4.5, Claude Opus 4.5
    - Google: Gemini 2.0 Flash, Gemini 3 Pro
    """

    # Model registry with (display_name, factory_function) tuples - populated as class attribute
    MODELS = {}

    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.model_instances = {}

    @classmethod
    def _register_models_class(cls):
        """Register all available models as class attributes."""
        # OpenAI Models
        cls.MODELS["gpt-4o"] = (
            "GPT-4o",
            lambda key: GPTModel(key, "gpt-4o", "GPT-4o")
        )
        cls.MODELS["gpt-5.1-instant"] = (
            "GPT-5.1",
            lambda key: GPTModel(key, "gpt-5.1", "GPT-5.1", reasoning_effort="none")
        )

        # Anthropic Models
        cls.MODELS["claude-sonnet-4.5"] = (
            "Claude Sonnet 4.5",
            lambda key: ClaudeModel(key, "claude-sonnet-4-5-20250929", "Claude Sonnet 4.5")
        )
        cls.MODELS["claude-opus-4.5-low"] = (
            "Claude Opus 4.5",
            lambda key: ClaudeModel(key, "claude-opus-4-5-20251101", "Claude Opus 4.5", thinking_budget=1024)
        )

        # Google Models
        cls.MODELS["gemini-2.0-flash"] = (
            "Gemini 2.0 Flash",
            lambda key: GeminiModel(key, "gemini-2.0-flash", "Gemini 2.0 Flash")
        )
        cls.MODELS["gemini-3-pro-low"] = (
            "Gemini 3 Pro",
            lambda key: GeminiThinkingModel(key, "gemini-3-pro-preview", "Gemini 3 Pro", thinking_level="low")
        )

    def _get_model_instance(self, model_key: str) -> ModelInterface:
        """Get or create a model instance."""
        if model_key not in self.model_instances:
            if model_key not in self.MODELS:
                raise ValueError(f"Unknown model: {model_key}. Available: {list(self.MODELS.keys())}")

            api_key = self._get_api_key_for_model(model_key)
            factory = self.MODELS[model_key][1]
            self.model_instances[model_key] = factory(api_key)

        return self.model_instances[model_key]

    def _get_api_key_for_model(self, model_key: str) -> str:
        """Determine which API key to use based on model key."""
        if model_key.startswith("claude"):
            return self.api_keys["anthropic"]
        elif model_key.startswith("gpt"):
            return self.api_keys["openai"]
        elif model_key.startswith("gemini"):
            return self.api_keys["google"]
        raise ValueError(f"Unknown model provider for: {model_key}")

    @staticmethod
    def generate_sample_client() -> ClientProfile:
        """Generate a random client profile from predefined templates."""
        templates = [
            ClientProfile(
                name="Alex",
                age=28,
                gender="male",
                presenting_problem="anxiety and perfectionism",
                context="Recently started a new job and feeling overwhelmed by performance expectations"
            ),
            ClientProfile(
                name="Sarah",
                age=35,
                gender="female",
                presenting_problem="depression and isolation",
                context="Working remotely for 2 years and struggling to maintain social connections"
            ),
        ]
        return random.choice(templates)

    @staticmethod
    def generate_sample_therapist() -> TherapistProfile:
        """Generate a random therapist profile from predefined approaches."""
        approaches = [
            TherapistProfile(
                approach="Cognitive Behavioral Therapy (CBT)",
                style="collaborative and solution-focused"
            ),
            TherapistProfile(
                approach="Person-Centered Therapy",
                style="empathetic and non-directive"
            ),
        ]
        return random.choice(approaches)

    def generate_session(
            self,
            client: Optional[ClientProfile] = None,
            therapist: Optional[TherapistProfile] = None,
            num_exchanges: int = 3,
            model: str = "claude-sonnet-4.5",
            max_tokens: int = 250,
            temperature: float = 0.7,
            prompt_config: Optional[PromptConfig] = None
    ) -> Generator[tuple[Dict[str, str], 'TherapySession'], None, None]:
        """
        Generate a therapy session with specified parameters.

        Args:
            client: Client profile (auto-generated if None)
            therapist: Therapist profile (auto-generated if None)
            num_exchanges: Number of therapist-client exchanges
            model: Model key (see MODELS dict for options)
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            prompt_config: Custom prompt configuration

        Yields:
            Tuple of (message dict, current session state)
        """
        client = client or self.generate_sample_client()
        therapist = therapist or self.generate_sample_therapist()
        prompt_config = prompt_config or PromptConfig()

        # Get display name for metadata
        display_name = self.MODELS[model][0]

        # Create metadata
        metadata = SessionMetadata(
            timestamp=datetime.now().isoformat(),
            model=display_name,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_config=prompt_config,
            client_profile=client,
            therapist_profile=therapist,
            num_exchanges=num_exchanges
        )
        conversation = []

        model_instance = self._get_model_instance(model)

        # Initial client sharing
        client_prompt = f"""{client.format_context(prompt_config.client_context_template)}

Share what brings you to therapy today:"""

        client_response = model_instance.generate_message(
            system=prompt_config.client_system,
            messages=[{"role": "user", "content": client_prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        message = {"role": "client", "content": client_response}
        conversation.append(message)
        yield message, TherapySession(metadata=metadata, conversation=conversation)

        # Generate conversation exchanges
        for _ in range(num_exchanges):
            # Therapist response
            convo_history = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}"
                                         for msg in conversation])

            therapist_prompt = f"""{therapist.format_context(prompt_config.therapist_context_template)}

Conversation so far:
{convo_history}

{prompt_config.therapist_instruction}"""

            therapist_response = model_instance.generate_message(
                system=prompt_config.therapist_system,
                messages=[{"role": "user", "content": therapist_prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            message = {"role": "therapist", "content": therapist_response}
            conversation.append(message)
            yield message, TherapySession(metadata=metadata, conversation=conversation)

            # Client response
            convo_history = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}"
                                         for msg in conversation])

            client_prompt = f"""{client.format_context(prompt_config.client_context_template)}

Conversation so far:
{convo_history}

{prompt_config.client_instruction}"""

            client_response = model_instance.generate_message(
                system=prompt_config.client_system,
                messages=[{"role": "user", "content": client_prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            message = {"role": "client", "content": client_response}
            conversation.append(message)
            yield message, TherapySession(metadata=metadata, conversation=conversation)


# Example usage
if __name__ == "__main__":
    # Example of how to use the updated generator
    api_keys = {
        "openai": "your-openai-key",
        "anthropic": "your-anthropic-key",
        "google": "your-google-key"
    }

    generator = TherapySessionGenerator(api_keys)

    # Print available models
    print("Available models:")
    for key, (display_name, _) in generator.MODELS.items():
        print(f"  {key}: {display_name}")

    # Example: Generate a session with Claude Opus 4.5 (low thinking)
    # for message, session in generator.generate_session(
    #     model="claude-opus-4.5-low",
    #     num_exchanges=2,
    #     max_tokens=300
    # ):
    #     print(f"\n{message['role'].upper()}: {message['content']}")

# Register models when module is imported
TherapySessionGenerator._register_models_class()