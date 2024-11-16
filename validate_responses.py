import argparse
import pandas as pd
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from query_data import query_rag
import logging

logger = logging.getLogger(__name__)

LOGFILE = "validate_responses.log"

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Do the task based on the following answer:

{answer}

---

Task: Output only the text "TRUE" if the response matches the answer, and only the text "FALSE" if the response doesnt not match the answer: {response}
"""



def main():
    logfile_handler = logging.FileHandler(LOGFILE)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile_handler.setFormatter(formatter)
    logger.addHandler(logfile_handler)
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="The file containing the answers and responses.")
    args = parser.parse_args()
    file_path = args.file
    df = pd.read_csv(file_path)
    
    count_matching = 0
    count_total = 0
    
    for index, row in df.iterrows():
        question = row['questions']
        answer = row['answers']
        response = query_rag(question, print_output=False)
        result = validate_response(answer, response)
        # log question, response, answer, correctness
        
        if result == "TRUE":
            result_bool = True
            count_matching += 1
        elif result == "FALSE":
            result_bool = False
        else:
            result_bool = None
            count_total -= 1
            
        count_total += 1
        
        logstring = f"""
        QUESTION INDEX: {index}
        QUESTION: {question}
        EXPECTED: {answer}
        RESPONSE: "{response}"
        MATCHES: {result}
        ============================
        """
        logger.info(logstring)
        
    accuracy = count_matching / count_total
    
    print(f"Accuracy: {accuracy}")



def validate_response(answer: str, response: str):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(answer=answer, response=response)
    
    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)
    
    return response_text
    
    
if __name__ == "__main__":
    main()