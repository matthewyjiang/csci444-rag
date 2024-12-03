# populate_database.py
import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding_function import initialize_embedding_function
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    global DATA_PATH
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Populate the Chroma vector database using TF-IDF weighted GloVe embeddings.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("data_path", type=str, help="The path to the data.")
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
        print("✅ Clearing Database")
        clear_database()

    # Load documents from the specified directory
    documents = load_documents()
    all_texts = [doc.page_content for doc in documents]

    # Initialize embedding function with the specified embedding_path and optional vectorizer_path
    embedding_function = initialize_embedding_function(
        documents=all_texts,
        embedding_path=args.embedding_path,
        tf_idf_path=None,
    )

    # Split documents into chunks
    chunks = split_documents(documents)

    # Add chunks to Chroma vector database
    add_to_chroma(chunks, embedding_function)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks, embedding_function):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_documents = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_documents.append(chunk)  # Append the entire Document object

    if new_documents:
        print(f"✅ Adding new documents: {len(new_documents)}")
        db.add_documents(new_documents)  # Chroma handles embedding internally
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page metadata.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()