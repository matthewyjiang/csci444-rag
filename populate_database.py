# populate_database.py
import argparse
import os
import shutil
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from embedding_function import initialize_embedding_function
from langchain.vectorstores import Chroma  # Import from langchain.vectorstores
import chromadb
from chromadb.config import Settings

# Define constants
CHROMA_PATH = "chroma"          # Directory to persist Chroma database
DATA_PATH = "data"              # Directory containing PDF documents
COLLECTION_NAME = "default"     # Name of the Chroma collection

def main():
    global DATA_PATH
    # Set up argument parser
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Populate the Chroma vector database using TF-IDF weighted GloVe embeddings.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("data_path", type=str, help="The path to the data.")
=======
    parser = argparse.ArgumentParser(
        description="Populate the Chroma vector database using TF-IDF weighted GloVe embeddings."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database by deleting the existing Chroma directory.",
    )
>>>>>>> 069b9b5 (relevance threshold)
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Path to the pre-trained word vectors file (e.g., word2vec).",
    )
    args = parser.parse_args()
    DATA_PATH = args.data_path
    # Reset the database if requested
    if args.reset:
        print("✅ Resetting the Chroma database...")
        clear_database()

    # Initialize Chroma client with updated Settings
    print("🗃️ Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)



    # Create or retrieve the collection with cosine similarity
    if COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        print(f"⚠️ Collection '{COLLECTION_NAME}' already exists.")
        collection = client.get_collection(name=COLLECTION_NAME)
        # Verify if the existing collection uses cosine similarity
        current_space = collection.get_metadata()["hnsw:space"]
        if current_space != "cosine":
            print(f"⚠️ Collection '{COLLECTION_NAME}' uses '{current_space}' instead of 'cosine'. Recreating the collection.")
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Recreated collection '{COLLECTION_NAME}' with cosine similarity.")
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' already uses cosine similarity.")
    else:
        print(f"🆕 Creating a new collection '{COLLECTION_NAME}' with cosine similarity.")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created with cosine similarity.")

    # Load documents from the specified directory
    print("📄 Loading documents...")
    documents = load_documents()
    all_texts = [doc.page_content for doc in documents]

    # Initialize embedding function with the specified embedding_path
    print("🔧 Initializing embedding function...")
    embedding_function = initialize_embedding_function(
        documents=all_texts,
        embedding_path=args.embedding_path,
        tf_idf_path=None,
    )

    # Split documents into chunks
    print("✂️ Splitting documents into chunks...")
    chunks = split_documents(documents)

    # Assign unique IDs to each chunk
    print("🔢 Assigning unique IDs to document chunks...")
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Initialize LangChain's Chroma with the ChromaDB client
    print("🗃️ Initializing LangChain's Chroma store...")
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        client=client
    )

    # Add documents to the collection
    print(f"📥 Adding {len(chunks_with_ids)} documents to the collection...")
    db.add_documents(documents=chunks_with_ids)
    # db.persist()
    print("✅ Documents added and Chroma database persisted.")

def load_documents():
    print(f"🔍 Loading PDFs from '{DATA_PATH}'...")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    print("🧩 Splitting documents using RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    """
    Assign unique IDs to each document chunk.
    Format: "source:page:chunk_index"
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown_source")
        page = chunk.metadata.get("page", "unknown_page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    """
    Deletes the Chroma persist directory to reset the database.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"🗑️ Deleted the existing Chroma directory at '{CHROMA_PATH}'.")
    else:
        print(f"ℹ️ No existing Chroma directory found at '{CHROMA_PATH}'.")

if __name__ == "__main__":
    main()