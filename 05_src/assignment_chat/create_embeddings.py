# create_embeddings.py
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env and .secrets
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '..', '.env')
secrets_path = os.path.join(script_dir, '..', '.secrets')
if os.path.exists(env_path):
    load_dotenv(env_path)
if os.path.exists(secrets_path):
    load_dotenv(secrets_path, override=True)

import pandas as pd
import chromadb

# Allow import from ../utils
sys.path.insert(0, os.path.join(script_dir, '..', 'utils'))
from clients import get_client

# Get the OpenAI client (using gateway if configured)
client = get_client()

# Initialize persistent ChromaDB client
db_client = chromadb.PersistentClient(path="./chroma_db")

# Collection name
collection_name = "travel_cities"

# Check if collection already exists
existing_collections = [c.name for c in db_client.list_collections()]
if collection_name in existing_collections:
    confirm = input(f"Collection '{collection_name}' already exists. Overwrite? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborting. Existing collection remains intact.")
        exit(0)
    else:
        db_client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'.")

# Create new collection (no embedding function)
collection = db_client.create_collection(name=collection_name)

# Load city descriptions
df = pd.read_csv("city_descriptions.csv")

# Generate embeddings for all descriptions
print("Generating embeddings...")
documents = df["description"].tolist()
embeddings = []
batch_size = 20
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch
    )
    embeddings.extend([d.embedding for d in resp.data])

# Prepare metadata and IDs
metadatas = [{"city": row.city, "country": row.country} for row in df.itertuples()]
ids = [f"city_{i}" for i in range(len(documents))]

# Add to collection
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"Added {len(documents)} city descriptions to ChromaDB.")