import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from embedding_function import load_embedding_function

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context/conceptual examples:

{context}

---

Answer the question consisely based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str, print_output=True):
    # Prepare the DB.
    embedding_function = load_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

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
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    if print_output:
        print(formatted_response)
    return response_text, sources


if __name__ == "__main__":
    main()
