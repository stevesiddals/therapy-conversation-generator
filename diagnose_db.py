# diagnose_db.py - updated version
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

# Analyze all documents
total_docs = conversations.count_documents({})
print(f"Total documents: {total_docs}")

# Structure patterns found
structures = {}
doc_count = 0

# Check each document
for doc in conversations.find({}):
    doc_count += 1
    doc_id = str(doc["_id"])

    # Define the structure signature
    has_metadata = "metadata" in doc
    has_prompt_config = has_metadata and "prompt_config" in doc["metadata"] if has_metadata else False

    # Check specific fields - only if we have prompt_config
    has_therapist_context = False
    has_client_context = False
    has_therapist_context_template = False
    has_client_context_template = False

    if has_prompt_config:
        prompt_config = doc["metadata"]["prompt_config"]
        has_therapist_context = "therapist_context" in prompt_config
        has_client_context = "client_context" in prompt_config
        has_therapist_context_template = "therapist_context_template" in prompt_config
        has_client_context_template = "client_context_template" in prompt_config

    # Create a signature of the document structure
    structure_sig = (
        has_metadata,
        has_prompt_config,
        has_therapist_context,
        has_client_context,
        has_therapist_context_template,
        has_client_context_template
    )

    # Track this structure
    if structure_sig not in structures:
        structures[structure_sig] = []
    structures[structure_sig].append(doc_id)

# Report the findings
print("\nDocument structure patterns found:")
for i, (struct, ids) in enumerate(structures.items()):
    print(f"\nStructure {i + 1}: (found in {len(ids)} documents)")
    print(f"  Has metadata: {struct[0]}")
    print(f"  Has prompt_config: {struct[1]}")
    print(f"  Has therapist_context: {struct[2]}")
    print(f"  Has client_context: {struct[3]}")
    print(f"  Has therapist_context_template: {struct[4]}")
    print(f"  Has client_context_template: {struct[5]}")
    print(f"  Example doc ID: {ids[0]}")

    # Try to show a sample of this structure
    try:
        sample_doc = conversations.find_one({"_id": ObjectId(ids[0])})
        if sample_doc is not None and "metadata" in sample_doc and "prompt_config" in sample_doc["metadata"]:
            print("\nPrompt config keys:")
            print(list(sample_doc["metadata"]["prompt_config"].keys()))
    except Exception as e:
        print(f"Error examining sample document: {e}")

print(f"\nProcessed {doc_count} of {total_docs} documents")

# Check specifically for documents with different structures
print("\nLooking for problematic documents...")
problem_structures = []
for struct, ids in structures.items():
    # If it has prompt_config but not the expected context fields
    if struct[1] and (not struct[2] or not struct[3]):
        problem_structures.extend(ids)
    # Or if it has the template fields instead of the regular fields
    elif struct[1] and (struct[4] or struct[5]):
        problem_structures.extend(ids)

if problem_structures:
    print(f"Found {len(problem_structures)} problematic documents:")
    for doc_id in problem_structures[:5]:  # Show at most 5
        try:
            doc = conversations.find_one({"_id": ObjectId(doc_id)})
            print(f"\nDocument ID: {doc_id}")
            if "metadata" in doc and "prompt_config" in doc["metadata"]:
                print("Prompt config keys:", list(doc["metadata"]["prompt_config"].keys()))
            else:
                print("Document structure:", list(doc.keys()))
        except Exception as e:
            print(f"Error examining problematic document {doc_id}: {e}")
else:
    print("No problematic documents found")