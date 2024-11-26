import argparse
import pandas as pd
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from query_data import query_rag
from validate_responses_openai import validate_response as validate_response_openai
import logging
import glob

logger = logging.getLogger(__name__)

LOGFILE = "validate_responses.log"

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a system designed to verify the accuracy of answers. Compare the RAG answer with the ground truth and determine if the RAG answer correctly reflects the ground truth. 

- RAG Answer: {RAG_ANSWER}
- Ground Truth: {GROUND_TRUTH}

If the RAG answer aligns with the ground truth, respond with "TRUE". If it does not align, respond with "FALSE". 

Respond with only one word: TRUE or FALSE.
"""



def main():
    logfile_handler = logging.FileHandler(LOGFILE)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile_handler.setFormatter(formatter)
    logger.addHandler(logfile_handler)
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="The directory containing the answers and responses.")
    args = parser.parse_args()
    file_dir = args.dir
    csv_files = glob.glob(file_dir + "/*.csv")
    dfs = []
    for file in csv_files:
        dfs.append((file, pd.read_csv(file)))
    
    count_matching = 0
    count_total = 0
    
    
    
    for file, df in dfs:
        logger.info(f"Validating responses in {file}")
        print (f"Validating responses in {file}")
        for index, row in df.iterrows():
            question = row['questions']
            answer = row['answers']
            
            logger.info("querying RAG + model")
            response, sources = query_rag(question, print_output=False)
            logger.info("querying validation model")
            result = validate_response_openai(answer, response.strip()).strip()

            
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
            SOURCES: {sources}
            MATCHES: {result_bool}
            ============================
            """
            logger.info(logstring)
            
        accuracy = count_matching / count_total
        
        print(f"Accuracy: {accuracy}")



def validate_response(answer: str, response: str):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(GROUND_TRUTH=answer, RAG_ANSWER=response)
    model = OllamaLLM(model="hermes3")
    response_text = model.invoke(prompt)
    
    return response_text
    
    
if __name__ == "__main__":
    main()