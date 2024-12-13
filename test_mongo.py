import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()


def test_mongodb_connection():
    # Replace this with your connection string
    connection_string = os.getenv("MONGODB_URI")

    if not connection_string:
        print("Error: MONGODB_URI not found in environment variables")
        return

    try:
        # Create client
        client = MongoClient(connection_string)

        # Test connection by listing database names
        client.list_database_names()

        print("Successfully connected to MongoDB!")

        # Test basic operations
        db = client.test_database
        collection = db.test_collection

        # Insert a test document
        result = collection.insert_one({"test": "document"})
        print(f"Inserted test document with id: {result.inserted_id}")

        # Find the document
        found = collection.find_one({"test": "document"})
        print(f"Found test document: {found}")

        # Clean up
        collection.delete_one({"test": "document"})
        print("Test document deleted")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
            print("Connection closed")


if __name__ == "__main__":
    test_mongodb_connection()