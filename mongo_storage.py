# mongo_storage.py
from pymongo import MongoClient
from datetime import datetime, UTC
from typing import List, Optional, Dict
from dataclasses import asdict
from bson.objectid import ObjectId

class MongoStorage:
    def __init__(self, connection_string: str, database_name: str = "therapy_conversations"):
        """Initialize MongoDB connection."""
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.conversations = self.db.conversations

    def save_therapy_session(self, session: 'TherapySession', researcher: Optional[str] = None) -> str:
        """Save a therapy session to MongoDB."""
        # Convert session to dictionary
        conversation_data = {
            "metadata": {
                "timestamp": session.metadata.timestamp,
                "model": session.metadata.model,
                "max_tokens": session.metadata.max_tokens,
                "temperature": session.metadata.temperature,
                "num_exchanges": session.metadata.num_exchanges,
                "prompt_config": asdict(session.metadata.prompt_config),
                "client_profile": asdict(session.metadata.client_profile),
                "therapist_profile": asdict(session.metadata.therapist_profile)
            },
            "conversation": session.conversation,
            "researcher": researcher,
            "created_at": datetime.utcnow()
        }

        # Insert into MongoDB
        result = self.conversations.insert_one(conversation_data)
        return str(result.inserted_id)

    def get_therapy_session(self, conversation_id: str) -> Optional['TherapySession']:
        """Retrieve a therapy session from MongoDB."""
        from therapy_simulator import (
            TherapySession, SessionMetadata, PromptConfig,
            ClientProfile, TherapistProfile
        )

        # Find conversation by ID
        from bson.objectid import ObjectId
        conversation_data = self.conversations.find_one({"_id": ObjectId(conversation_id)})

        if not conversation_data:
            return None

        # Reconstruct objects
        prompt_config = PromptConfig(**conversation_data["metadata"]["prompt_config"])
        client_profile = ClientProfile(**conversation_data["metadata"]["client_profile"])
        therapist_profile = TherapistProfile(**conversation_data["metadata"]["therapist_profile"])

        metadata = SessionMetadata(
            timestamp=conversation_data["metadata"]["timestamp"],
            model=conversation_data["metadata"]["model"],
            max_tokens=conversation_data["metadata"]["max_tokens"],
            temperature=conversation_data["metadata"]["temperature"],
            prompt_config=prompt_config,
            client_profile=client_profile,
            therapist_profile=therapist_profile,
            num_exchanges=conversation_data["metadata"]["num_exchanges"]
        )

        return TherapySession(
            metadata=metadata,
            conversation=conversation_data["conversation"]
        )

    def get_average_rating(self, conversation_id: str) -> Optional[float]:
        """Calculate the average rating for a conversation.

        Returns:
            Optional[float]: Average rating on a 1-5 scale, or None if no ratings
        """
        feedback_list = self.get_feedback(conversation_id)
        ratings = []

        rating_values = {
            'very_negative': 1,
            'negative': 2,
            'neutral': 3,
            'positive': 4,
            'very_positive': 5
        }

        for feedback in feedback_list:
            if 'rating' in feedback and feedback['rating'] in rating_values:
                ratings.append(rating_values[feedback['rating']])

        return sum(ratings) / len(ratings) if ratings else None

    def list_conversations(self, limit: int = 100, skip: int = 0) -> List[Dict]:
        """List conversations with complete metadata."""
        cursor = self.conversations.find(
            {},
            {
                # Keep the same projection as before
                "metadata.timestamp": 1,
                "metadata.model": 1,
                "metadata.max_tokens": 1,
                "metadata.temperature": 1,
                "metadata.num_exchanges": 1,
                "metadata.client_profile.name": 1,
                "metadata.client_profile.age": 1,
                "metadata.client_profile.gender": 1,
                "metadata.client_profile.presenting_problem": 1,
                "metadata.client_profile.context": 1,
                "metadata.therapist_profile.approach": 1,
                "metadata.therapist_profile.style": 1,
                "metadata.prompt_config.therapist_system": 1,
                "metadata.prompt_config.client_system": 1,
                "metadata.prompt_config.therapist_context": 1,
                "metadata.prompt_config.client_context": 1,
                "metadata.prompt_config.therapist_instruction": 1,
                "metadata.prompt_config.client_instruction": 1,
                "researcher": 1,
                "created_at": 1,
                "feedback": 1
            }
        ).sort("created_at", -1).skip(skip).limit(limit)

        conversations = []
        for doc in cursor:
            conv_dict = {
                # Basic identifiers
                "id": str(doc["_id"]),
                "timestamp": doc["metadata"]["timestamp"],
                "researcher": doc.get("researcher"),
                "created_at": doc.get("created_at"),

                # Client info
                "client_name": doc["metadata"]["client_profile"]["name"],
                "age": doc["metadata"]["client_profile"]["age"],
                "gender": doc["metadata"]["client_profile"]["gender"],
                "presenting_problem": doc["metadata"]["client_profile"]["presenting_problem"],
                "context": doc["metadata"]["client_profile"]["context"],

                # Therapist info
                "approach": doc["metadata"]["therapist_profile"]["approach"],
                "style": doc["metadata"]["therapist_profile"]["style"],

                # Group model parameters under metadata
                "metadata": {
                    "model": doc["metadata"]["model"],
                    "max_tokens": doc["metadata"]["max_tokens"],
                    "temperature": doc["metadata"]["temperature"],
                    "num_exchanges": doc["metadata"]["num_exchanges"]
                },

                # Prompt configuration
                "prompt_config": {
                    "therapist_system": doc["metadata"]["prompt_config"]["therapist_system"],
                    "client_system": doc["metadata"]["prompt_config"]["client_system"],
                    "therapist_context": doc["metadata"]["prompt_config"]["therapist_context"],
                    "client_context": doc["metadata"]["prompt_config"]["client_context"],
                    "therapist_instruction": doc["metadata"]["prompt_config"]["therapist_instruction"],
                    "client_instruction": doc["metadata"]["prompt_config"]["client_instruction"]
                }
            }

            # Get average rating and feedback count
            conv_dict['average_rating'] = self.get_average_rating(str(doc['_id']))
            conv_dict['feedback_count'] = len(doc.get('feedback', []))

            conversations.append(conv_dict)

        return conversations

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from MongoDB."""
        from bson.objectid import ObjectId
        result = self.conversations.delete_one({"_id": ObjectId(conversation_id)})
        return result.deleted_count > 0

    def get_stats(self) -> Dict:
        """Get basic statistics about stored conversations."""
        return {
            "total_conversations": self.conversations.count_documents({}),
            "unique_researchers": len(self.conversations.distinct("researcher")),
            "unique_approaches": len(self.conversations.distinct("metadata.therapist_profile.approach")),
            "average_exchanges": self.conversations.aggregate([
                {"$group": {
                    "_id": None,
                    "avg_exchanges": {"$avg": "$metadata.num_exchanges"}
                }}
            ]).next()["avg_exchanges"]
        }

    def update_researcher(self, conversation_id: str, new_researcher: str) -> bool:
        """Update the researcher name for a conversation."""
        from bson.objectid import ObjectId
        result = self.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"researcher": new_researcher}}
        )
        return result.modified_count > 0

    def add_feedback(self, conversation_id: str, researcher_name: str, comment: str,
                     rating: Optional[str] = None) -> bool:
        """Add feedback to a conversation.

        Args:
            conversation_id: The ID of the conversation
            researcher_name: Name of the researcher providing feedback
            comment: The feedback comment
            rating: Optional rating (positive/negative/neutral)

        Returns:
            bool: True if feedback was added successfully
        """
        feedback_entry = {
            "researcher_name": researcher_name,
            "timestamp": datetime.now(UTC),
            "comment": comment,
            "rating": rating
        }

        result = self.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$push": {"feedback": feedback_entry}}
        )
        return result.modified_count > 0

    def get_feedback(self, conversation_id: str) -> List[Dict]:
        """Get all feedback for a conversation.

        Args:
            conversation_id: The ID of the conversation

        Returns:
            List of feedback entries
        """
        conversation = self.conversations.find_one(
            {"_id": ObjectId(conversation_id)},
            {"feedback": 1}
        )

        return conversation.get("feedback", []) if conversation else []

    def delete_feedback(self, conversation_id: str, feedback_timestamp: datetime) -> bool:
        """Delete a specific feedback entry from a conversation.

        Args:
            conversation_id: The ID of the conversation
            feedback_timestamp: Timestamp of the feedback to delete

        Returns:
            bool: True if feedback was deleted successfully
        """
        result = self.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$pull": {"feedback": {"timestamp": feedback_timestamp}}}
        )
        return result.modified_count > 0

    def get_feedback_count(self, conversation_id: str) -> int:
        """Get the number of feedback entries for a conversation.

        Args:
            conversation_id: The ID of the conversation

        Returns:
            int: Number of feedback entries
        """
        conversation = self.conversations.find_one(
            {"_id": ObjectId(conversation_id)},
            {"feedback": 1}
        )

        return len(conversation.get("feedback", [])) if conversation else 0