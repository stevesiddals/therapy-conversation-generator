# migrate_field_names.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()
mongodb_uri = os.getenv("MONGODB_URI")

# Connect to MongoDB
client = MongoClient(mongodb_uri)
db = client["therapy_conversations"]
conversations = db.conversations

# Find all documents that use the old field names
docs_to_update = conversations.find({
    "metadata.prompt_config.therapist_context": {"$exists": True}
})

update_count = 0
for doc in docs_to_update:
    doc_id = doc["_id"]

    # Get the current values
    prompt_config = doc["metadata"]["prompt_config"]
    therapist_context = prompt_config.get("therapist_context", "")
    client_context = prompt_config.get("client_context", "")

    # Update document with new field names while preserving the values
    result = conversations.update_one(
        {"_id": doc_id},
        {
            "$set": {
                "metadata.prompt_config.therapist_context_template": therapist_context,
                "metadata.prompt_config.client_context_template": client_context
            },
            "$unset": {
                "metadata.prompt_config.therapist_context": "",
                "metadata.prompt_config.client_context": ""
            }
        }
    )

    if result.modified_count > 0:
        update_count += 1

print(f"Updated {update_count} documents")

# Verify all documents now use the new field names
old_field_count = conversations.count_documents({
    "$or": [
        {"metadata.prompt_config.therapist_context": {"$exists": True}},
        {"metadata.prompt_config.client_context": {"$exists": True}}
    ]
})

new_field_count = conversations.count_documents({
    "$and": [
        {"metadata.prompt_config.therapist_context_template": {"$exists": True}},
        {"metadata.prompt_config.client_context_template": {"$exists": True}}
    ]
})

print(f"Documents with old field names remaining: {old_field_count}")
print(f"Documents with new field names: {new_field_count}")
print(f"Total documents: {conversations.count_documents({})}")

print("Migration complete")