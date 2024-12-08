import os

import json
import random
import anthropic
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ClientProfile:
    name: str
    age: int
    gender: str
    presenting_problem: str
    context: str

    def to_prompt(self) -> str:
        return f"""You are speaking directly as a therapy client with these characteristics:
- Name: {self.name}
- Age: {self.age}
- Gender: {self.gender}
- Main challenge: {self.presenting_problem}
- Current situation: {self.context}

Express your thoughts and feelings naturally, as yourself. Share honestly while maintaining appropriate boundaries. 
If asked about suicide or self-harm, always indicate that while you're struggling, you have no such thoughts or plans.

Speak in first person and respond directly without mentioning your name or role."""


@dataclass
class TherapistProfile:
    approach: str
    style: str

    def to_prompt(self) -> str:
        return f"""You are speaking directly as a therapist using {self.approach} with a {self.style} style.

Guidelines:
- Show empathy and genuine care while maintaining professional boundaries
- Use techniques appropriate to your therapeutic approach
- Focus on understanding the client's experience
- Help identify patterns and potential growth areas
- Maintain safety and refer to crisis resources if needed
- Keep responses concise and focused (100-200 words)

Speak in first person and respond directly without mentioning your role."""


@dataclass
class SessionMetadata:
    timestamp: str
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str
    client_profile: ClientProfile
    therapist_profile: TherapistProfile
    num_exchanges: int

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
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

    def _extract_text_content(self, response) -> str:
        """Extract and combine all text content from response blocks."""
        text_content = []
        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
        return " ".join(text_content).strip()

    def generate_session(self,
                         client: Optional[ClientProfile] = None,
                         therapist: Optional[TherapistProfile] = None,
                         num_exchanges: int = 3,
                         model: str = "claude-3-sonnet-20240229",
                         max_tokens: int = 150,
                         temperature: float = 0.7,
                         system_prompt: str = "You are an AI helping to simulate therapy conversations for research.") -> TherapySession:
        """Generate a therapy session with specified parameters."""

        client = client or self.generate_sample_client()
        therapist = therapist or self.generate_sample_therapist()

        # Create metadata
        metadata = SessionMetadata(
            timestamp=datetime.now().isoformat(),
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            client_profile=client,
            therapist_profile=therapist,
            num_exchanges=num_exchanges
        )

        conversation = []
        context = f"""This is a therapy conversation.

Client context: {client.to_prompt()}

Therapist context: {therapist.to_prompt()}

Keep the conversation natural and focused. Each response should be 2-4 sentences."""

        # Initial client sharing
        client_prompt = f"{context}\n\nShare what brings you to therapy today:"
        client_response = self._extract_text_content(
            self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
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
            therapist_prompt = f"{context}\n\nConversation so far:\n{convo_history}\n\nRespond as the therapist:"

            therapist_response = self._extract_text_content(
                self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
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
            client_prompt = f"{context}\n\nConversation so far:\n{convo_history}\n\nRespond as yourself:"

            client_response = self._extract_text_content(
                self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": client_prompt}]
                )
            )

            conversation.append({
                "role": "client",
                "content": client_response
            })

        return TherapySession(metadata=metadata, conversation=conversation)


def print_conversation(session: TherapySession) -> None:
    """Pretty print a therapy session including metadata and conversation."""
    print("\n=== Session Metadata ===")
    print(f"Timestamp: {session.metadata.timestamp}")
    print(f"Model: {session.metadata.model}")
    print(f"Temperature: {session.metadata.temperature}")
    print(f"Max Tokens: {session.metadata.max_tokens}")
    print(f"Number of Exchanges: {session.metadata.num_exchanges}")

    print("\n=== Client Profile ===")
    client = session.metadata.client_profile
    print(f"Name: {client.name}")
    print(f"Age: {client.age}")
    print(f"Gender: {client.gender}")
    print(f"Presenting Problem: {client.presenting_problem}")
    print(f"Context: {client.context}")

    print("\n=== Therapist Profile ===")
    therapist = session.metadata.therapist_profile
    print(f"Approach: {therapist.approach}")
    print(f"Style: {therapist.style}")

    print("\n=== Conversation ===")
    for message in session.conversation:
        role = message["role"].capitalize()
        content = message["content"].strip()
        print(f"\n{role}: {content}")