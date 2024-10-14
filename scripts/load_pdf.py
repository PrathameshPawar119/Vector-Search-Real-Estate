import openai
import os
import time
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.schema import Document
from pymongo import MongoClient
from dotenv import load_dotenv
import pdfplumber  # To extract text from the PDF

# Load environment variables
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

client = MongoClient(os.getenv('MONGO_URI'))
db = client[os.getenv('MONGO_DB_NAME')]
collection = db[os.getenv('MONGO_COLLECTION_NAME_2_4_3_v1')]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine"
)

def load_pdf_pages(file_path, num_pages=100):
    """
    Load the PDF and yield each page as a chunk.
    """
    with pdfplumber.open(file_path) as pdf:
        for page_num in range(0, min(num_pages, len(pdf.pages))):  # Limit to 100 pages
            page = pdf.pages[page_num]
            text = page.extract_text()
            if text.strip():
                yield page_num, text

def generate_questions_answers(page_content):
    prompt = f"Based on the following text, generate a set of important questions that could be asked and provide answers:\n\n{page_content}"
    
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",  # Adjust the model as per your requirement
        messages=[
            {"role": "system", "content": "You are a knowledge assistant that generates questions and answers from provided content (return only json)."},
            {"role": "user", "content": prompt}
        ]
    )
    
    response = completion.choices[0].message.content
    return response

def chatbot(page_content, query):
    prompt = f"Answer this question :{query},\n  Based on the following text, :\n\n{page_content}"
    
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": "You are a knowledge assistant that answers questions based on provided content."},
            {"role": "user", "content": prompt}
        ]
    )
    
    response = completion.choices[0].message.content
    return response

def embed_and_store_pages(file_path, num_pages=100):
    documents = []
    
    # Load each page and process
    for page_num, page_content in load_pdf_pages(file_path, num_pages):
        print(f"Processing page {page_num+1}...")
        
        # Generate questions and answers based on page content
        qa_content = generate_questions_answers(page_content)
        
        # Combine the original page content with generated Q&A for embedding
        combined_content = f"Original Text:\n{page_content}\n\nQuestions & Answers:\n{qa_content}"
        
        # Create embedding for the combined content
        embedding = embeddings_model.embed_query(combined_content)
        
        if embedding:
            doc = Document(
                page_content=combined_content,
                metadata={"page_num": page_num+1}
            )
            documents.append(doc)
        
        # Throttle API requests to avoid hitting rate limits
        time.sleep(1)

        # Insert in batches of 10
        if len(documents) % 10 == 0:
            vectorstore.add_documents(documents)
            documents = []
    
    # Insert any remaining documents
    if documents:
        vectorstore.add_documents(documents)

if __name__ == "__main__":
    file_path = "./data/spring-boot-reference 2.4.3.pdf"
    embed_and_store_pages(file_path, num_pages=100)

    question = "Features of spring boot?"
    results = vectorstore.similarity_search(question, k=1)
    response = chatbot(results, question)
    
    print(question, results)
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")

    print("\n\n")
    print("Response from chatbot:\n ", response)
