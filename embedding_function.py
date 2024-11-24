# embedding_function.py

import os
import numpy as np
import nltk
from typing import List, Dict
from langchain.embeddings.base import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import KeyedVectors

# Ensure the punkt tokenizer is downloaded
nltk.download('punkt')
default_embedding_path: str = "word_vectors.joblib"
default_tf_idf_path: str = "tfidf_vectorizer.joblib"
class EmbeddingFunction(Embeddings):
    def __init__(self, embedding_path: str, tf_idf_path: str = None):
        """
        Initializes the EmbeddingFunction by loading pre-trained word vectors
        and setting up the TF-IDF vectorizer.

        Parameters:
        - embedding_path (str): Path to the pre-trained Word2Vec word vectors file.
        - tf_idf_path (str, optional): Path to load a pre-fitted TF-IDF vectorizer.
        """
        self.embedding_dim = 300  # Assuming Word2Vec 300d
        self.word_vectors = self.load_word_vectors(embedding_path)
        print(f"✅ Loaded word vectors from '{embedding_path}'. Embedding dimension: {self.embedding_dim}")

        if tf_idf_path and os.path.exists(tf_idf_path):
            self.vectorizer = self.load_vectorizer(tf_idf_path)
            print(f"✅ Loaded TF-IDF vectorizer from '{tf_idf_path}'.")
        else:
            self.vectorizer = TfidfVectorizer()
            print("🛠️ Initialized a new TF-IDF vectorizer.")

    def load_word_vectors(self, filepath: str) -> KeyedVectors:
        """
        Loads Word2Vec word vectors using gensim's KeyedVectors.

        Parameters:
        - filepath (str): Path to the Word2Vec word vectors file.

        Returns:
        - KeyedVectors: Gensim's KeyedVectors instance.
        """
        print("🔄 Loading Word2Vec word vectors using gensim...")
        try:
            word_vectors = KeyedVectors.load_word2vec_format(filepath, binary=False)
            print("✅ Word2Vec word vectors loaded successfully.")
            return word_vectors
        except Exception as e:
            print(f"❌ Failed to load word vectors using gensim: {e}")
            raise

    def load_vectorizer(self, filepath: str) -> TfidfVectorizer:
        """
        Loads a pre-fitted TF-IDF vectorizer from disk.

        Parameters:
        - filepath (str): Path to the saved TF-IDF vectorizer.

        Returns:
        - TfidfVectorizer: Loaded TF-IDF vectorizer.
        """
        print(f"🔄 Loading TF-IDF vectorizer from '{filepath}'...")
        return joblib.load(filepath)

    def fit_vectorizer(self, documents: List[str]):
        """
        Fits the TF-IDF vectorizer on the provided documents.

        Parameters:
        - documents (List[str]): List of document texts.
        """
        print("🔄 Fitting TF-IDF vectorizer on the documents...")
        self.vectorizer.fit(documents)
        print("✅ TF-IDF vectorizer fitted successfully.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds multiple documents using TF-IDF weighted averaging of word vectors.

        Parameters:
        - texts (List[str]): List of document texts.

        Returns:
        - List[List[float]]: List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_sentence(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query using TF-IDF weighted averaging of word vectors.

        Parameters:
        - text (str): The query text.

        Returns:
        - List[float]: Embedding vector.
        """
        return self.embed_sentence(text)

    def embed_sentence(self, sentence: str) -> List[float]:
        """
        Embeds a sentence by weighted averaging of its word vectors based on TF-IDF scores.
        """
        tokens = nltk.word_tokenize(sentence.lower())
        tfidf_scores = self.vectorizer.transform([sentence]).toarray()[0]
        feature_names = self.vectorizer.get_feature_names_out()
        word_to_tfidf = {word: tfidf_scores[idx] for idx, word in enumerate(feature_names)}

        vectors = []
        weights = []
        for word in tokens:
            if word in self.word_vectors and word in word_to_tfidf:
                vectors.append(self.word_vectors[word])
                weights.append(word_to_tfidf[word])

        if not vectors:
            return [0.0] * self.embedding_dim

        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1, 1)
        weighted_vectors = vectors * weights
        sentence_embedding = np.sum(weighted_vectors, axis=0) / np.sum(weights)
        return sentence_embedding.tolist()

    def save_models(self, save_embedding_path: str = default_embedding_path, save_tf_idf_path: str = default_tf_idf_path):
        """
        Saves the word vectors and TF-IDF vectorizer to disk.

        Parameters:
        - save_embedding_path (str): File path to save the word vectors.
        - save_tf_idf_path (str): File path to save the TF-IDF vectorizer.
        """
        # Save word vectors
        joblib.dump(self.word_vectors, save_embedding_path)
        print(f"✅ Word vectors saved to '{save_embedding_path}'.")

        # Save TF-IDF vectorizer
        joblib.dump(self.vectorizer, save_tf_idf_path)
        print(f"✅ TF-IDF vectorizer saved to '{save_tf_idf_path}'.")

    def load_models(self, load_embedding_path: str = default_embedding_path, load_tf_idf_path: str = default_tf_idf_path):
        """
        Loads the word vectors and TF-IDF vectorizer from disk.

        Parameters:
        - load_embedding_path (str): File path to load the word vectors.
        - load_tf_idf_path (str): File path to load the TF-IDF vectorizer.

        Returns:
        - None
        """
        # Load word vectors
        if os.path.exists(load_embedding_path):
            self.word_vectors = joblib.load(load_embedding_path)
            self.embedding_dim = len(next(iter(self.word_vectors.values())))
            print(f"✅ Word vectors loaded from '{load_embedding_path}'. Embedding dimension: {self.embedding_dim}")
        else:
            raise FileNotFoundError(f"❌ Word vectors file '{load_embedding_path}' not found.")

        # Load TF-IDF vectorizer
        if os.path.exists(load_tf_idf_path):
            self.vectorizer = joblib.load(load_tf_idf_path)
            print(f"✅ TF-IDF vectorizer loaded from '{load_tf_idf_path}'.")
        else:
            raise FileNotFoundError(f"❌ TF-IDF vectorizer file '{load_tf_idf_path}' not found.")


def initialize_embedding_function(documents: List[str], embedding_path: str, tf_idf_path: str = None) -> EmbeddingFunction:
    """
    Initializes the embedding function with the provided documents and word vectors.

    Parameters:
    - documents (List[str]): List of text documents.
    - embedding_path (str): Path to the pre-trained Word2Vec word vectors file.
    - tf_idf_path (str, optional): Path to a pre-fitted TF-IDF vectorizer.

    Returns:
    - EmbeddingFunction: Initialized EmbeddingFunction instance.
    """
    embedding = EmbeddingFunction(embedding_path, tf_idf_path)

    if tf_idf_path and os.path.exists(tf_idf_path):
        # Vectorizer is already loaded
        pass
    else:
        # Fit the vectorizer on the documents
        embedding.fit_vectorizer(documents)
        print("✅ TF-IDF vectorizer fitted on the documents.")
        # Save the fitted TF-IDF vectorizer
        if tf_idf_path:
            embedding.save_models(save_tf_idf_path=tf_idf_path)
        else:
            embedding.save_models()

    return embedding


def load_embedding_function(embedding_path: str, tf_idf_path: str = default_tf_idf_path) -> EmbeddingFunction:
    """
    Loads the embedding function by loading pre-trained word vectors and TF-IDF vectorizer.

    Parameters:
    - embedding_path (str): Path to the pre-trained Word2Vec word vectors file.
    - tf_idf_path (str): Path to the pre-fitted TF-IDF vectorizer.

    Returns:
    - EmbeddingFunction: Loaded EmbeddingFunction instance.
    """
    embedding = EmbeddingFunction(embedding_path, tf_idf_path)
    return embedding


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Initialize the embedding function using custom word vectors and TF-IDF weighting.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the text files (each page as a document).")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to the pre-trained Word2Vec word vectors file.")
    parser.add_argument("--save_embedding_path", type=str, default=default_embedding_path, help="Path to save the word vectors.")
    parser.add_argument("--save_tf_idf_path", type=str, default=default_tf_idf_path, help="Path to save the TF-IDF vectorizer.")
    args = parser.parse_args()

    # Gather all text file paths from the input directory
    input_directory = args.input_dir
    pdf_text_paths = [
        os.path.join(input_directory, fname)
        for fname in os.listdir(input_directory)
        if fname.endswith('.txt')
    ]

    if not pdf_text_paths:
        print("❌ No text files found in the specified directory.")
        exit(1)

    # Read and collect all document texts
    documents = []
    for path in pdf_text_paths:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if content:  # Ensure non-empty content
                documents.append(content)

    # Initialize embedding function
    embedding_func = initialize_embedding_function(documents, args.embedding_path, args.save_tf_idf_path)
    print("✅ Embedding function initialized.")

    # Save the models
    embedding_func.save_models(save_embedding_path=args.save_embedding_path, save_tf_idf_path=args.save_tf_idf_path)