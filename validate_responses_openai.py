from openai import OpenAI
from query_data import query_rag
import logging


PROMPT_TEMPLATE = """
You are a system designed to verify the accuracy of answers. Compare the RAG answer with the ground truth and determine if the RAG answer correctly reflects the ground truth. 

- RAG Answer: {RAG_ANSWER}
- Ground Truth: {GROUND_TRUTH}

If the RAG answer aligns with the ground truth, respond with "TRUE". If it does not align, respond with "FALSE". Be lenient, allow for answers that mean close to the same thing as the ground truth. 

Respond with only one word: TRUE or FALSE.
"""
client = OpenAI()


def validate_response(answer: str, response: str):
    prompt_template = PROMPT_TEMPLATE
    prompt = prompt_template.format(RAG_ANSWER=answer, GROUND_TRUTH=response)
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
    