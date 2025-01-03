import pdfplumber
from dotenv import load_dotenv
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import numpy as np

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")


pc = Pinecone(api_key=pinecone_api_key)

# PDF Parsing
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def summarize_text(text):
    url = "https://api.groq.ai/v1/summarize"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model": "groq-gpt-4",
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        summary = response.json().get('summary', '')
        return summary.strip() if summary else "No summary available."
    except requests.exceptions.RequestException as e:
        print(f"Error summarizing text: {e}")
        return "Error generating summary."

def split_text_into_chunks(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)

def create_pinecone_index(index_name, chunks):
    embeddings = None

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )


    index = pc.Index(index_name)


    embeddings_data = np.random.rand(len(chunks), 1536)

    vectors = [(str(i), embedding.tolist(), {"text": chunk}) for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_data))]

    index.upsert(vectors)

    return index

def retrieve_chunks(query, vector_store):

    query_embedding = np.random.rand(1, 1536)

    # Updated query method with keyword arguments
    results = vector_store.query(vector=query_embedding.tolist(), top_k=3)

    return " ".join([match['metadata']['text'] for match in results['matches']])