# Real Estate Vector Search API

## Overview
This project implements a vector search-based real estate recommendation system using MongoDB, OpenAI embeddings, and Flask. It allows users to search for properties using natural language queries, leveraging vector similarity to find relevant listings and providing AI-enhanced responses.

## Features
- Natural language property search using vector embeddings
- AI-powered response generation for property recommendations
- MongoDB Atlas vector search integration
- RESTful API endpoint for property queries

## Tech Stack
- Python 3.8+
- Flask (Web framework)
- MongoDB Atlas (Database with vector search capability)
- OpenAI API (for embeddings and response generation)

## Prerequisites
- Python 3.8 or higher
- MongoDB Atlas account with vector search enabled
- OpenAI API key

## Project Structure
```
/vector_search_project
│
├── /app
│   ├── __init__.py
│   ├── embeddings.py     # Handles embedding generation
│   ├── db.py             # Database connection and operations
│   └── api.py            # Flask API endpoints
│
├── /data
│   └── dataset.csv       # Real estate dataset
│
├── /scripts
│   └── load_data.py      # Script to load and embed data
│
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── app.py               # Main application entry point
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vector_search_project.git
cd vector_search_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
MONGO_URI=your_mongodb_connection_string
```

## Data Loading and Embedding

Before running the API, you need to load and embed the real estate data:

1. Ensure your dataset is in the correct format and placed in `data/dataset.csv`
2. Run the data loading script:
```bash
python scripts/load_data.py
```

This script will:
- Load the real estate data
- Generate embeddings for each property
- Store the data and embeddings in MongoDB
- Create the necessary vector search index
- [create a custom vector search index if necessary]([create custom vector search index](https://www.mongodb.com/developer/products/atlas/using-openai-latest-embeddings-rag-system-mongodb/#step-6--create-a-vector-search-index))

## Running the Application

Start the Flask application:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Usage

### Vector Search Endpoint

**Endpoint:** `POST /vector_search`

**Request Body:**
```json
{
  "query": "3 bedroom house in Aguadilla under $200,000"
}
```

**Response:**
```json
{
  "response": "Detailed AI-generated response about matching properties",
  "source_information": "Information about the properties used to generate the response"
}
```

## Example Queries

1. Basic location and bedroom query:
```json
{
  "query": "3 bedroom houses in Aguadilla"
}
```

2. Price range query:
```json
{
  "query": "homes under $150,000 in San Juan"
}
```

3. Complex feature query:
```json
{
  "query": "large houses with more than 2000 square feet and a pool"
}
```

## Technical Details

### Vector Search Implementation

The system uses the following pipeline for vector search:
```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding_vector",
            "numCandidates": 150,
            "limit": 5
        }
    },
    {
        "$project": {
            "_id": 0,
            "brokered_by": 1,
            "status": 1,
            "price": 1,
            # ... other fields
        }
    }
]
```

### Embedding Generation

Properties are embedded using OpenAI's text-embedding-3-small model. The embedding input combines various property features:
```python
embedding_input = f"{property['brokered_by']}, {property['status']}, Price: {property['price']}, Beds: {property['bed']}, ..."
```

## Troubleshooting

Common issues and solutions:

1. **No results returned**: 
   - Verify that the vector index is created correctly
   - Check if documents have embedding vectors
   - Ensure query embedding dimensionality matches document embeddings

2. **MongoDB connection issues**: 
   - Verify your MongoDB URI in the .env file
   - Ensure your IP is whitelisted in MongoDB Atlas

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the embedding and language models
- MongoDB for their vector search capability
