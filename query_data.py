# query_data.py
import argparse
import os
import sys
from typing import List
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from groq import Groq  # Ensure Groq is correctly imported
from langchain_ollama import OllamaLLM

from embedding_function import load_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context/conceptual examples:

{context}

---

Answer the question based on the above context: {question}
"""

def initialize():
    parser = argparse.ArgumentParser(description="Interactive Query the Chroma vector database using Word2Vec and TF-IDF embeddings.")
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Path to the Word2Vec word vectors file.",
    )
    
    args = parser.parse_args()

    print("🔧 Loading Word2Vec and TF-IDF Embedding Function")
    embedding_function = load_embedding_function(
        embedding_path=args.embedding_path,
    )

    print("🗃️ Initializing Chroma Database")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("✅ Chroma Database initialized and ready.")

    return db

def query_llm(prompt: str):
    print("📄 Generated Prompt:\n", prompt)

    # Initialize the Groq client
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Generate the response using the Groq client
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

def process_query(db: Chroma, query_text: str):
    print(f"\n🔍 Processing Query: {query_text}")

    # Perform the similarity search
    results = db.similarity_search_with_score(query_text, k=5)

    # Compile context from the results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query the LLM
    chat_completion = query_llm(prompt)
    response_text = chat_completion.choices[0].message.content

    # Extract sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    formatted_response = f"\n\n&&&Response: {response_text}\n\nSources: {sources}"
    print("📝 Formatted Response:\n", formatted_response)
    
def query_rag(db: Chroma, query_text: str):
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    # print (results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)
    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    return response_text, sources

def main():
    db = initialize()
    print("\n🚀 Enter your queries below. Type 'exit' or 'quit' to terminate.")

    while True:
        try:
            query_text = input("\nYour Query: ").strip()
            if query_text.lower() in {'exit', 'quit'}:
                print("👋 Exiting the query application.")
                break
            elif not query_text:
                print("⚠️ Please enter a valid query.")
                continue

            process_query(db, query_text)

        except KeyboardInterrupt:
            print("\n👋 Received KeyboardInterrupt. Exiting.")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()