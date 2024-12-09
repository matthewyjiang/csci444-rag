import argparse
import os
from typing import List
from langchain.vectorstores import Chroma  # Import from langchain.vectorstores
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from groq import Groq  # Ensure Groq is correctly installed
from langchain_ollama import OllamaLLM  # Ensure OllamaLLM is correctly installed

import chromadb
from chromadb.config import Settings

from embedding_function import load_embedding_function

CHROMA_PATH = "chroma"          # Must match populate_database.py
COLLECTION_NAME = "default"     # Must match populate_database.py

PROMPT_TEMPLATE = """
Answer the question based only on the following context/conceptual examples:

{context}

---

Use the above contexts to answer this question: {question}
"""

def initialize():
    parser = argparse.ArgumentParser(
        description="Interactive Query the Chroma vector database using Word2Vec and TF-IDF embeddings."
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Path to the Word2Vec word vectors file.",
    )
    
    args = parser.parse_args()

    print("🔧 Loading Word2Vec and TF-IDF Embedding Function...")
    embedding_function = load_embedding_function(
        embedding_path=args.embedding_path,
    )

    print("🗃️ Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)


    print("🗃️ Initializing LangChain's Chroma store...")
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        client=client
    )

    print("✅ Chroma Database initialized and ready.")
    return db

def query_llm(prompt: str):
    # print("📄 Generated Prompt:\n", prompt)

    # Initialize the Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable not set.")
        return "Error: GROQ_API_KEY not set.", []

    client = Groq(
        api_key=groq_api_key,
    )

    # Generate the response using the Groq client
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion
    except Exception as e:
        print(f"❌ Error querying LLM: {e}")
        return "Error querying LLM.", []

def main():
    db = initialize()
    print("\n🚀 Enter your queries below. Type 'exit' or 'quit' to terminate.")

    while True:
        try:
            query_text = input("\n❓Your Query: ").strip()
            if query_text.lower() in {'exit', 'quit'}:
                print("👋 Exiting the query application.")
                break
            elif not query_text:
                print("⚠️ Please enter a valid query.")
                continue

            # Set a reasonable threshold for similarity scores
            threshold = 0.5
            response_text, sources = query_rag(db, query_text, threshold)

            print("📝 Response:\n", response_text)
            print("🔗 Sources:\n", sources)

        except KeyboardInterrupt:
            print("\n👋 Received Keyboard Interrupt. Exiting.")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

def query_rag(db: Chroma, query_text: str, threshold: float = 0.3):
    """
    Perform a similarity search and query the LLM with the results that meet the similarity score threshold.

    Parameters:
    - db (Chroma): The Chroma database instance.
    - query_text (str): The query text.
    - threshold (float): The similarity score threshold.

    Returns:
    - response_text (str): The response from the LLM.
    - sources (List[str]): The sources of the context used for the response.
    """
    # Perform the similarity search
    results = db.similarity_search_with_score(query_text, k=5)

    # Print the similarity scores to analyze their range
    print("Similarity Scores:")
    for doc, score in results:
        print(f"*️⃣ Score: {score}, Document ID: {doc.metadata.get('id', 'N/A')}, Page_content: {doc.page_content}\n")

    # Filter results based on the threshold
    filtered_results = [(doc, score) for doc, score in results if score <= threshold]

    if not filtered_results:
        return "No relevant results found based on the similarity threshold.", []

    # Compile context from the filtered results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query the LLM
    response = query_llm(prompt)
    if isinstance(response, str):
        response_text = response
    else:
        response_text = response.choices[0].message.content

    # Extract sources
    sources = [doc.metadata.get("id", None) for doc, _score in filtered_results]

    return response_text, sources

if __name__ == "__main__":
    main()