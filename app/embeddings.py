import openai
from dotenv import load_dotenv
import os
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = "text-embedding-3-small" 

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""
    try:
        response = openai.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
