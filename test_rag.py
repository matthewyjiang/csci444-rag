import argparse
from query_data import query_rag
from langchain_ollama import OllamaLLM

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, help="The file containing the test questions.")
    args = parser.parse_args()
    test_file = args.test_file

    with open(test_file, "r") as file:
        questions = file.readlines()

    is_tf_file = test_file.endswith("_TF.txt")
    results = []
    for count, line in enumerate(questions):
        if line.strip():
            print(f"Processing question {count + 1}...")
            expected_response, question = parse_question(line)
            if is_tf_file:
                question += " (Start response with your answer: True/False, then short explanation <30 words)"
            response_text = query_rag(question)
            is_correct = evaluate_response(response_text, expected_response)
            print(f"Correct: {is_correct}")
            results.append((question, expected_response, response_text, is_correct))

    with open("responses.txt", "w") as file:
        for count, (question, expected_response, response_text, is_correct) in enumerate(results, start=1):            
            file.write(f"{count}.\n")
            file.write(f"Question: {question}\n")
            file.write(f"Expected Response: {expected_response}\n")
            file.write(f"Response: {response_text}\n")
            file.write(f"Correct: {is_correct}\n\n")
        
        print(f"Score: {sum(is_correct for _, _, is_correct in results)}")
        file.write(f"Score: {sum(is_correct for _, _, is_correct in results)} \n")

def parse_question(line):
    parts = line.split(": ", 1)
    expected_response = parts[0].lower().replace("•", "").strip() == "true"
    print("Expected response: ", expected_response)
    question = parts[1].strip()
    return expected_response, question

def evaluate_response(response_text, expected_response):
    
    prompt = EVAL_PROMPT.format(
        expected_response="true" if expected_response else "false",
        actual_response=response_text
    )

    model = OllamaLLM(model="llama3.2")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    if "true" in evaluation_results_str_cleaned:
        return True
    elif "false" in evaluation_results_str_cleaned:
        return False
    else:
        response_text += f"Ambiguous response. Cannot determine if 'true' or 'false'."
        return False

if __name__ == "__main__":
    main()