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

LOGFILE = "logs/relevant1.log"

CHROMA_PATH = "chroma"

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
    count_relevant = 0
    count_relevant_and_matching = 0
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
            result = validate_response_openai(answer, response.strip(), question).strip()

            result = result.split()
            
            if "TRUE" in result:
                result_bool = True
                count_matching += 1
            elif "FALSE" in result:
                result_bool = False
            
            if "RELEVANT" in result:
                count_relevant += 1
                relevant_bool = True
            elif "IRRELEVANT" in result:
                relevant_bool = False
                
                
            if relevant_bool and result_bool:
                count_relevant_and_matching += 1
                
            count_total += 1
            
            logstring = f"""
            QUESTION INDEX: {index}
            QUESTION: {question}
            EXPECTED: {answer}
            RESPONSE: "{response}"
            SOURCES: {sources}
            MATCHES: {result_bool}
            RELEVANT_SOURCES: {relevant_bool}
            ============================
            """
            logger.info(logstring)
            
        accuracy = count_matching / count_total
        
        relevancy = count_relevant / count_total
        
        relevant_given_matching = count_relevant_and_matching / count_matching
        matching_given_relevant = count_relevant_and_matching / count_relevant
        
        print(f"Accuracy: {accuracy}")
        print(f"Relevancy: {relevancy}")
        print(f"Relevant given Matching: {relevant_given_matching}")
        print(f"Matching given Relevant: {matching_given_relevant}")
        
        if file.split("/")[-1].split("_")[1][0] == "t":
            questiontype = "tf"
        else:
            questiontype = "sa"
        
        fileid = file.split("/")[-1].split(".")[0][:3] + questiontype
        
        print(f"csv: \n{fileid},temp,{accuracy},{relevancy},{relevant_given_matching},{matching_given_relevant}")
        



def validate_response(answer: str, response: str, question: str) -> str:
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(GROUND_TRUTH=answer, RAG_ANSWER=response, QUESTION=question)
    model = OllamaLLM(model="hermes3")
    response_text = model.invoke(prompt)
    
    return response_text
    
    
if __name__ == "__main__":
    main()