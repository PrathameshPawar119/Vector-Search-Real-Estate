import pymongo
from dotenv import load_dotenv
import os

load_dotenv()

def get_collection():
    # Load MongoDB URI from environment variables
    mongo_uri = os.getenv('MONGO_URI')
    client = pymongo.MongoClient(mongo_uri)
    db = client['real_estate']
    collection = db['properties']
    return collection


def verify_database_setup():
    collection = get_collection()
    
    # Check document count
    doc_count = collection.count_documents({})
    print(f"Number of documents in collection: {doc_count}")
    
    # Check a sample document
    sample = collection.find_one()
    if sample:
        print("Sample document fields:", list(sample.keys()))
        if 'embedding_vector' in sample:
            print(f"Embedding vector length: {len(sample['embedding_vector'])}")
        else:
            print("WARNING: No embedding_vector found in sample document")
    
    # List indexes
    indexes = list(collection.list_indexes())
    print("Indexes:", [index['name'] for index in indexes])