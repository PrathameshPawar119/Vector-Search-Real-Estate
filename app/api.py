from flask import Flask, request, jsonify
from app.db import get_collection, verify_database_setup
from app.embeddings import get_embedding
import openai
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')

def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.
    """
    print(f"Received query: {user_query}")
    verify_database_setup()
    
    query_embedding = get_embedding(user_query)
    print(f"Generated embedding of length: {len(query_embedding)}")
    
    # Verify a document in the collection
    sample_doc = collection.find_one()
    if sample_doc:
        print("Sample document structure:", sample_doc.keys())
        if 'embedding_vector' in sample_doc:
            print(f"Sample embedding vector length: {len(sample_doc['embedding_vector'])}")
    else:
        print("No documents found in collection")

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding_vector",
                "numCandidates": 150,
                "limit": 5  # Increased from 3 to 5 for more results
            }
        },
        {
            "$addFields": {
                "_id": 0,  # Exclude the _id field
                "brokered_by": 1,  # Include the brokered_by field
                "status": 1,  # Include the status field
                "price": 1,  # Include the price field
                "bed": 1,  # Include the bed field
                "bath": 1,  # Include the bath field
                "acre_lot": 1,  # Include the acre lot field
                "street": 1,  # Include the street field
                "city": 1,  # Include the city field
                "state": 1,  # Include the state field
                "zip_code": 1,  # Include the zip code field
                "house_size": 1,  # Include the house size field
                "prev_sold_date": 1,  # Include the previous sold date field
                "score": {
                    "$meta": "vectorSearchScore"  # Include the search score
                }
            }
        }
    ]


    try:
        print("Executing pipeline...")
        results = list(collection.aggregate(pipeline))
        print(f"Number of results: {len(results)}")
        if not results:
            print("No results found")
            return "No matching properties found."
        # print("\nResults:\n", results)
        return results
    except Exception as e:
        print(f"Vector search error: {e}")
        return f"Error performing vector search: {str(e)}"


def handle_user_query(query, collection):
    """
    Handle user query by performing vector search and generating AI-driven responses.
    """
    search_results = vector_search(query, collection)

    if isinstance(search_results, str):  # Check for error message
        return search_results, ""

    completion = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are a Real Estate recommendation system. Provide detailed responses based on the available property information."},
            {"role": "user", "content": f"Based on the following properties, answer this query: {query}\n\nAvailable properties:\n{search_results}"}
        ]
    )

    return completion.choices[0].message.content, search_results


def verify_vector_index(collection):
    indexes = collection.list_indexes()
    vector_index_exists = False
    for index in indexes:
        if 'vector_index' in index['name']:
            vector_index_exists = True
            break
    
    if not vector_index_exists:
        print("Vector index not found. Creating index...")
        collection.create_search_index(
            "vector_index",
            definition={
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding_vector": {
                            "dimensions": 1536,  # Adjust if using a different embedding size
                            "similarity": "cosine",
                            "type": "knnVector"
                        }
                    }
                }
            }
        )
        print("Vector index created successfully!")
    else:
        print("Vector index exists!")

@app.route("/vector_search", methods=["POST"])
def vector_search_endpoint():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    collection = get_collection()

    # Handle the user query and get the response from AI
    response, source_information = handle_user_query(user_query, collection)

    return jsonify({"response": response, "source_information": source_information})
