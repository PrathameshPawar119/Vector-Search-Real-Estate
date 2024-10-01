
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.join(current_dir, '..'))

# Now import the required modules
from app.embeddings import get_embedding

from app.db import get_collection
from app.api import verify_vector_index
from app.api import vector_search

def test_vector_search():
    collection = get_collection()
    
    # Verify index
    verify_vector_index(collection)
    
    # Test queries
    test_queries = [
        "houses with 3 bedrooms in Aguadilla",
        "properties under $150,000",
        "large homes with more than 2000 square feet"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        results = vector_search(query, collection)
        
        if isinstance(results, list):
            print(f"Found {len(results)} results")
            for result in results[:2]:  # Print first 2 results
                print(f"- Price: ${result.get('price', 'N/A')}, "
                      f"Bedrooms: {result.get('bed', 'N/A')}, "
                      f"City: {result.get('city', 'N/A')}")
        else:
            print(f"Error or no results: {results}")

if __name__ == "__main__":
    test_vector_search()