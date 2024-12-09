# therapy_simulator.py
import json
import random
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import anthropic


@dataclass
class PromptConfig:
    therapist_system: str = "You are a therapist."
    client_system: str = "You are a client in a therapy session."

    therapist_context: str = """You are practicing {approach}, with a {style} style. 
Remain compassionate and validating, providing a safe space for the client to explore their experiences. 
What follows is the therapy conversation so far."""

    client_context: str = """You are {name}, {age} years old and {gender}. 
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
    def __init__(self, api_key: str):
        self.client = anthropic.Client(api_key=api_key)

    def _extract_text_content(self, response) -> str:
        """Extract and combine all text content from response blocks."""
        text_content = []
        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
        return " ".join(text_content).strip()

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
            model: str = "claude-3-sonnet-20240229",
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

        # Initial client sharing
        client_prompt = f"""{client.format_context(prompt_config.client_context)}

Share what brings you to therapy today:"""

        client_response = self._extract_text_content(
            self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=prompt_config.client_system,
                messages=[{"role": "user", "content": client_prompt}]
            )
        )

        conversation.append({
            "role": "client",
            "content": client_response
        })

        # Generate conversation exchanges
        for _ in range(num_exchanges):
            # Therapist response
            convo_history = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}"
                                         for msg in conversation])

            therapist_prompt = f"""{therapist.format_context(prompt_config.therapist_context)}

Conversation so far:
{convo_history}

{prompt_config.therapist_instruction}"""

            therapist_response = self._extract_text_content(
                self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=prompt_config.therapist_system,
                    messages=[{"role": "user", "content": therapist_prompt}]
                )
            )

            conversation.append({
                "role": "therapist",
                "content": therapist_response
            })

            # Client response
            convo_history = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}"
                                         for msg in conversation])

            client_prompt = f"""{client.format_context(prompt_config.client_context)}

Conversation so far:
{convo_history}

{prompt_config.client_instruction}"""

            client_response = self._extract_text_content(
                self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=prompt_config.client_system,
                    messages=[{"role": "user", "content": client_prompt}]
                )
            )

            conversation.append({
                "role": "client",
                "content": client_response
            })

        return TherapySession(metadata=metadata, conversation=conversation)