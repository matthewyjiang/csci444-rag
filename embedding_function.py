# embedding_function.py

import os
import joblib
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from typing import List
from langchain.embeddings.base import Embeddings  # Ensure LangChain is installed

# Define paths for saving models and config
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
PCA_PATH = "pca_model.joblib"

class EmbeddingFunction(Embeddings):
    def __init__(self, max_features=256, embedding_dim=128):
        """
        Initializes the EmbeddingFunction with TF-IDF Vectorizer and PCA.
    
        Parameters:
        - max_features (int): Maximum number of features for TF-IDF.
        - embedding_dim (int): Number of dimensions for PCA.
        """
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.pca = PCA(n_components=self.embedding_dim)
    
    def fit(self, documents: List[str]):
        """
        Fits the TF-IDF vectorizer and PCA on the provided documents.
    
        Parameters:
        - documents (list of str): List of text documents.
        """
        # Fit TF-IDF Vectorizer
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        print("TF-IDF Vectorization complete.")
        
        # Convert sparse matrix to dense
        tfidf_dense = tfidf_matrix.toarray()
        
        # Normalize TF-IDF vectors
        tfidf_normalized = normalize(tfidf_dense)
        
        # Fit PCA on normalized TF-IDF vectors
        self.pca.fit(tfidf_normalized)
        print("PCA fitting complete.")
    
    def transform(self, documents: List[str]) -> 'numpy.ndarray':
        """
        Transforms documents into embedding vectors using fitted TF-IDF and PCA.
    
        Parameters:
        - documents (list of str): List of text documents.
    
        Returns:
        - embeddings (numpy.ndarray): Array of embedding vectors.
        """
        # Transform documents using TF-IDF Vectorizer
        tfidf_matrix = self.vectorizer.transform(documents)
        tfidf_dense = tfidf_matrix.toarray()
        tfidf_normalized = normalize(tfidf_dense)
        
        # Transform using PCA
        embeddings = self.pca.transform(tfidf_normalized)
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds multiple documents.
        
        Parameters:
        - texts (List[str]): List of document texts.
        
        Returns:
        - List[List[float]]: List of embedding vectors.
        """
        embeddings = self.transform(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query.
        
        Parameters:
        - text (str): The query text.
        
        Returns:
        - List[float]: Embedding vector.
        """
        embedding = self.transform([text])
        return embedding[0].tolist()
    
    def save_models(self, vectorizer_path=VECTORIZER_PATH, pca_path=PCA_PATH):
        """
        Saves the fitted TF-IDF vectorizer and PCA model to disk.
    
        Parameters:
        - vectorizer_path (str): File path to save the TF-IDF vectorizer.
        - pca_path (str): File path to save the PCA model.
        """
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.pca, pca_path)
        print(f"Models saved to '{vectorizer_path}' and '{pca_path}'.")
    
    def load_models(self, vectorizer_path=VECTORIZER_PATH, pca_path=PCA_PATH):
        """
        Loads the TF-IDF vectorizer and PCA model from disk.
    
        Parameters:
        - vectorizer_path (str): File path to load the TF-IDF vectorizer.
        - pca_path (str): File path to load the PCA model.
        """
        if not (os.path.exists(vectorizer_path) and os.path.exists(pca_path)):
            raise FileNotFoundError("Model files not found. Please fit the models first.")
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.pca = joblib.load(pca_path)
        print(f"Models loaded from '{vectorizer_path}' and '{pca_path}'.")

def initialize_embedding_function(documents: List[str]) -> EmbeddingFunction:
    """
    Initializes and fits the embedding function on the provided documents.
    
    Parameters:
    - documents (list of str): List of text documents.
    
    Returns:
    - embedding_function (EmbeddingFunction): Fitted EmbeddingFunction instance.
    """
    embedding = EmbeddingFunction()
    print(f"Total documents/pages to embed: {len(documents)}")
    
    # Fit the embedding function
    embedding.fit(documents)
    
    # Save the models
    embedding.save_models()
    
    return embedding

def load_embedding_function() -> EmbeddingFunction:
    """
    Loads the embedding function by loading saved TF-IDF vectorizer and PCA model.
    
    Returns:
    - embedding_function (EmbeddingFunction): Loaded EmbeddingFunction instance.
    """
    embedding = EmbeddingFunction()
    embedding.load_models()
    return embedding

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize and fit the embedding function using TF-IDF.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the text files (each page as a document).")
    
    args = parser.parse_args()
    
    # Gather all text file paths from the input directory
    input_directory = args.input_dir
    pdf_text_paths = [os.path.join(input_directory, fname) for fname in os.listdir(input_directory) if fname.endswith('.txt')]
    
    if not pdf_text_paths:
        print("No text files found in the specified directory.")
        exit(1)
    
    # Read and collect all document texts
    documents = []
    for path in pdf_text_paths:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if content:  # Ensure non-empty content
                documents.append(content)
    
    # Initialize and fit the embedding function
    embedding_func = initialize_embedding_function(documents)
    print("Embedding function initialized and models saved.")