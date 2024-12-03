from openai import OpenAI
from query_data import query_rag
import logging

PROMPT_TEMPLATE = """
You are a system designed to verify the accuracy of answers and the relevance of sources.

Compare the RAG answer with the ground truth and determine if the RAG answer correctly reflects the ground truth. Determine if the sources provided are relevant to the question.
Question: {QUESTION}
RAG Answer: {RAG_ANSWER}
Ground Truth: {GROUND_TRUTH}
The sources are prepended to the question.

If the RAG answer aligns with the ground truth, respond with "TRUE" or "FALSE" for accuracy.
If the sources are relevant to the question, respond with "RELEVANT" or "IRRELEVANT" for source relevance.

Respond with two words: one for accuracy (TRUE/FALSE) and one for source relevance (RELEVANT/IRRELEVANT), in that order.
"""
client = OpenAI()


def validate_response(answer: str, response: str, question: str) -> str:
    prompt_template = PROMPT_TEMPLATE
    prompt = prompt_template.format(RAG_ANSWER=answer, GROUND_TRUTH=response, QUESTION=question)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a part of a response validation pipeline."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    response = completion.choices[0].message.content
    return response
    