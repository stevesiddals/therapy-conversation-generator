# therapy_simulator.py

import random
from openai import OpenAI
import google.generativeai as genai
import anthropic
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

class ModelInterface(ABC):
    @abstractmethod
    def generate_message(self, system: str, messages: list, max_tokens: int, temperature: float) -> str:
        pass

class ClaudeModel(ModelInterface):
    MODEL_NAME = "claude-3-7-sonnet-latest"
    DISPLAY_NAME = "Anthropic Claude 3.7 Sonnet"

    def __init__(self, api_key: str):
        self.client = anthropic.Client(api_key=api_key)

    def generate_message(self, system: str, messages: list, max_tokens: int, temperature: float) -> str:
        response = self.client.messages.create(
            model=self.MODEL_NAME,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return self._extract_text_content(response)

    def _extract_text_content(self, response) -> str:
        text_content = []
        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
        return " ".join(text_content).strip()

class GPT4Model(ModelInterface):
    MODEL_NAME = "gpt-4o-2024-08-06"
    DISPLAY_NAME = "OpenAI ChatGPT-4o"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_message(self, system: str, messages: list, max_tokens: int, temperature: float) -> str:
        openai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


class GeminiModel(ModelInterface):
    MODEL_NAME = "gemini-2.0-flash"
    DISPLAY_NAME = "Google Gemini 2.0 Flash"

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
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
    MODELS = {
        GPT4Model.MODEL_NAME: (GPT4Model.DISPLAY_NAME, GPT4Model),
        ClaudeModel.MODEL_NAME: (ClaudeModel.DISPLAY_NAME, ClaudeModel),
        GeminiModel.MODEL_NAME: (GeminiModel.DISPLAY_NAME, GeminiModel)
    }

    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.model_instances = {}

    def _get_model_instance(self, model_name: str) -> ModelInterface:
        if model_name not in self.model_instances:
            api_key = self._get_api_key_for_model(model_name)
            model_class = self.MODELS[model_name][1]
            self.model_instances[model_name] = model_class(api_key)
        return self.model_instances[model_name]

    def _get_api_key_for_model(self, model_name: str) -> str:
        if model_name.startswith("claude"):
            return self.api_keys["anthropic"]
        elif model_name.startswith("gpt"):
            return self.api_keys["openai"]
        elif model_name.startswith("gemini"):
            return self.api_keys["google"]
        raise ValueError(f"Unknown model: {model_name}")

    def generate_sample_client(self) -> ClientProfile:
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

    def generate_sample_therapist(self) -> TherapistProfile:
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
            model: str = "claude-3-sonnet",
            max_tokens: int = 250,
            temperature: float = 0.7,
            prompt_config: Optional[PromptConfig] = None
    ) -> TherapySession:
        """Generate a therapy session with specified parameters."""

        client = client or self.generate_sample_client()
        therapist = therapist or self.generate_sample_therapist()
        prompt_config = prompt_config or PromptConfig()

        # Create metadata
        metadata = SessionMetadata(
            timestamp=datetime.now().isoformat(),
            model=model,
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
#        print_message = {"role": "client", "content": f"""Client template: {prompt_config.client_context_template}\n\nClient prompt: {client_prompt}\n\nClient response: {client_response}"""}
        conversation.append(message)
        yield message, TherapySession(metadata=metadata,
                                      conversation=conversation)  # Yield both message and current session

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
#            print_message = {"role": "therapist",
#                       "content": f"""Therapist prompt: {therapist_prompt}\n\nTherapist response: {therapist_response}"""}
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
#           print_message = {"role": "client",
#                            "content": f"""Client template: {prompt_config.client_context_template}\n\nClient prompt: {client_prompt}\n\nClient response: {client_response}"""}
            conversation.append(message)
            yield message, TherapySession(metadata=metadata, conversation=conversation)
