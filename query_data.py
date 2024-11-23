import argparse
import os
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from groq import Groq  # Import Groq

from embedding_function import load_embedding_function  # Updated import

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context/conceptual examples:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    # chat_completion = query_llm(query_text)
    # response_text = chat_completion.choices[0].message.content
    # formatted_response = f"\n\n&&&Response: {response_text}"
    # print(formatted_response)
    prompt, results = create_query_with_context(query_text)
    query_rag(prompt, results)

def create_query_with_context(query_text: str):
    embedding_function = load_embedding_function()  # Updated function call
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt, results

def query_llm(prompt: str):
    print(prompt)

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

def query_rag(prompt: str, results):
    print(prompt)

    # Initialize the Groq client
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Generate the response using the Groq client
    chat_completion = query_llm(prompt)

    # Access the response text correctly
    response_text = chat_completion.choices[0].message.content

    # print(f"&&&chat_completion:\n\n{chat_completion}")
    # print(f"&&&response_text:\n\n{response_text}")
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\n\n&&&Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()