# get_embedding_function.py
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
import json
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Define paths for saving models and config
VECTORIZER_PATH = "vectorizer.joblib"
PCA_PATH = "pca.joblib"
EMBEDDING_MODEL_PATH = "embedding_model.pth"
CONFIG_PATH = "embedding_config.json"

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SimpleEmbeddingModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.embedding_layer(x)

class EmbeddingWrapper:
    def __init__(self, input_dim, embedding_dim):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.model = SimpleEmbeddingModel(input_dim, embedding_dim)
        self.pca = PCA(n_components=embedding_dim)
        self.vectorizer = CountVectorizer(max_features=input_dim)  # Limit vocabulary size

    def fit_pca(self, embeddings):
        self.pca.fit(embeddings)

    def reduce_dimensionality(self, embeddings):
        return self.pca.transform(embeddings)

    def embed_query(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_tensor = torch.tensor(query_vector, dtype=torch.float32)
        with torch.no_grad():
            embeddings = self.model(query_tensor).numpy()
        reduced_embeddings = self.reduce_dimensionality(embeddings)
        return reduced_embeddings[0].tolist()  # Convert to list

    def embed_documents(self, docs):
        # Transform documents to vectors
        docs_vector = self.vectorizer.transform(docs).toarray()
        docs_tensor = torch.tensor(docs_vector, dtype=torch.float32)
        
        # Pass through the embedding model
        with torch.no_grad():
            embeddings = self.model(docs_tensor).numpy()
        
        # Reduce dimensionality using PCA
        reduced_embeddings = self.reduce_dimensionality(embeddings)
        
        return reduced_embeddings.tolist()  # Convert to list

    def save(self):
        # Save Vectorizer
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        # Save PCA
        joblib.dump(self.pca, PCA_PATH)
        # Save Embedding Model
        torch.save(self.model.state_dict(), EMBEDDING_MODEL_PATH)
        # Save configuration
        config = {
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f)

    def load(self):
        if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(PCA_PATH) and os.path.exists(EMBEDDING_MODEL_PATH) and os.path.exists(CONFIG_PATH)):
            raise FileNotFoundError("Embedding components not found. Please initialize them first.")
        
        # Load configuration
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        self.input_dim = config["input_dim"]
        self.embedding_dim = config["embedding_dim"]
        
        # Initialize components
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.pca = joblib.load(PCA_PATH)
        self.model = SimpleEmbeddingModel(self.input_dim, self.embedding_dim)
        self.model.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, weights_only=True))
        self.model.eval()

def initialize_embedding_function(all_documents):
    input_dim = 256  # Power of 2s
    embedding_dim_default = 256

    # Initialize vectorizer and fit
    vectorizer = CountVectorizer(max_features=input_dim)
    vectorizer.fit(all_documents)

    n_features = len(vectorizer.get_feature_names_out())
    n_samples = len(all_documents)
    embedding_dim = min(embedding_dim_default, n_features, n_samples)

    if embedding_dim < embedding_dim_default:
        print(f"✅ Embedding dimension reduced to {embedding_dim} due to data size constraints.")

    # Initialize EmbeddingWrapper with adjusted embedding_dim
    wrapper = EmbeddingWrapper(input_dim, embedding_dim)
    wrapper.vectorizer = vectorizer

    # Initialize embedding model with the correct dimensions
    wrapper.model = SimpleEmbeddingModel(input_dim, embedding_dim)
    wrapper.model.eval()  # Set to evaluation mode

    # Transform documents and embed
    all_vectorized_data = wrapper.vectorizer.transform(all_documents).toarray()
    all_tensor = torch.tensor(all_vectorized_data, dtype=torch.float32)
    with torch.no_grad():
        all_embeddings = wrapper.model(all_tensor).numpy()

    # Fit PCA on the embeddings
    wrapper.fit_pca(all_embeddings)

    # Save the embedding components
    wrapper.save()

    return wrapper

def load_embedding_function():
    wrapper = EmbeddingWrapper(input_dim=256, embedding_dim=256)  # Placeholder values; will be overwritten by loaded config
    wrapper.load()  # This sets the correct input_dim and embedding_dim
    # wrapper = OllamaEmbeddings(model="nomic-embed-text")
    return wrapper