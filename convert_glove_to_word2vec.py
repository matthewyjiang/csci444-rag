# convert_glove_to_word2vec.py
from gensim.scripts.glove2word2vec import glove2word2vec
import os

def convert_glove_to_word2vec(glove_input_path, word2vec_output_path):
    if not os.path.exists(glove_input_path):
        print(f"❌ GloVe file '{glove_input_path}' does not exist.")
        return
    try:
        glove2word2vec(glove_input_path, word2vec_output_path)
        print(f"✅ Successfully converted '{glove_input_path}' to '{word2vec_output_path}'.")
    except Exception as e:
        print(f"❌ Failed to convert GloVe to Word2Vec format: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert GloVe vectors to Word2Vec format.")
    parser.add_argument("--glove_input", type=str, required=True, help="Path to the GloVe input file.")
    parser.add_argument("--word2vec_output", type=str, required=True, help="Path to save the Word2Vec formatted output file.")
    args = parser.parse_args()

    convert_glove_to_word2vec(args.glove_input, args.word2vec_output)