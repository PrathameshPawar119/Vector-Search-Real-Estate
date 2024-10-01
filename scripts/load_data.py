import pandas as pd
import openai
from dotenv import load_dotenv
import os
import sys
import time

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.join(current_dir, '..'))

# Now import the required modules
from app.embeddings import get_embedding
from app.db import get_collection

load_dotenv()

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def batch_insert_documents(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        collection = get_collection()
        collection.insert_many(batch)
        time.sleep(0.2)  # Add a 1-second delay between API calls
        print(f"Inserted batch {i//batch_size + 1} out of {len(documents)//batch_size + 1}")


def load_and_prepare_data():
    print("Loading dataset...")
    df = pd.read_csv('data/realtor-data.csv')

    # Data cleaning and preparation
    print("Cleaning and preparing data...")
    df.fillna('', inplace=True)  # Fill NaN values with an empty string for all columns

    # Combine relevant columns to create a comprehensive description for embeddings
    df['embedding_input'] = (
        df['brokered_by'].astype(str) + ', ' +
        df['status'].astype(str) + ', ' +
        'Price: ' + df['price'].astype(str) + ', ' +
        'Beds: ' + df['bed'].astype(str) + ', ' +
        'Baths: ' + df['bath'].astype(str) + ', ' +
        'Acre Lot: ' + df['acre_lot'].astype(str) + ', ' +
        df['street'].astype(str) + ', ' +
        df['city'].astype(str) + ', ' +
        df['state'].astype(str) + ', ' +
        'ZIP: ' + df['zip_code'].astype(str) + ', ' +
        'House Size: ' + df['house_size'].astype(str) + ', ' +
        'Prev Sold Date: ' + df['prev_sold_date'].astype(str)
    )

    # Step 4: Create embeddings for each entry
    print("Generating embeddings...")
    embeddings = []
    for idx, data in enumerate(df['embedding_input']):
        print(f"Generating embedding for row {idx+1}/{len(df)}: {data}")
        embedding = get_embedding(data)
        if embedding is None:
            print(f"Failed to generate embedding for row {idx+1}")
        embeddings.append(embedding)

        # Throttle API requests to avoid rate limits
        time.sleep(0.2)  # Add a 1-second delay between API calls

    df['embedding_vector'] = embeddings

    # Prepare the data to be inserted into MongoDB
    documents = df.to_dict(orient='records')

    # Step 5: Batch insert documents into the database
    print("Inserting data into MongoDB in batches...")
    batch_insert_documents(documents)
    print("Data insertion complete. ooohhhoo 😍")

if __name__ == "__main__":
    load_and_prepare_data()
