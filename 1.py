"""
STAR Model Embedding Extraction - Cell 1
This cell extracts embeddings from the STAR model and saves them for later use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import pickle
import warnings

warnings.filterwarnings('ignore')

class STARModel:
    """STAR Model wrapper for embedding extraction"""

    def __init__(self, model_name='AIDA-UPM/star'):
        """
        Initialize STAR model

        Args:
            model_name (str): Hugging Face model name
        """
        print("Loading STAR model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

        # Freeze all STAR model parameters for fine-tuning
        self._freeze_star_parameters()

    def _freeze_star_parameters(self):
        """
        Freeze all STAR model parameters to prevent them from being updated during fine-tuning.
        Only the task-specific Siamese network will be trained.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        print("STAR model parameters frozen for fine-tuning")

    def extract_embeddings(self, texts, batch_size=32):
        """
        Extract style embeddings from texts

        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for processing

        Returns:
            np.ndarray: Style embeddings (n_texts, 1024)
        """
        embeddings = []
        
        print(f"Extracting embeddings for {len(texts)} texts...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i+batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


def load_stylometric_dataset(csv_path='/kaggle/input/testing/good_dataset_truncate.csv'):
    """
    Load the stylometric dataset from CSV file
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    authors = df['author'].tolist()
    
    unique_authors = list(set(authors))
    author_to_id = {author: idx for idx, author in enumerate(unique_authors)}
    author_ids = [author_to_id[author] for author in authors]
    
    print(f"Dataset loaded: {len(texts)} texts from {len(unique_authors)} authors")
    print(f"Authors: {unique_authors}")

    return texts, author_ids, unique_authors, author_to_id


def save_embeddings_and_metadata(embeddings, author_ids, unique_authors, author_to_id, 
                                save_path='/kaggle/working/'):
    """
    Save embeddings and metadata to files
    """
    import os
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save embeddings as numpy array
    embeddings_path = os.path.join(save_path, 'star_embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save metadata as pickle
    metadata = {
        'author_ids': author_ids,
        'unique_authors': unique_authors,
        'author_to_id': author_to_id,
        'num_texts': len(author_ids),
        'num_authors': len(unique_authors),
        'embedding_dim': embeddings.shape[1]
    }
    
    metadata_path = os.path.join(save_path, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")
    
    # Also save as JSON for easy inspection
    import json
    metadata_json = metadata.copy()
    metadata_json['author_ids'] = author_ids  # Keep as list for JSON
    
    json_path = os.path.join(save_path, 'metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata_json, f, indent=2)
    print(f"Metadata also saved as JSON to: {json_path}")
    
    return embeddings_path, metadata_path


def main_extraction():
    """Main function for embedding extraction"""
    print("STAR Model Embedding Extraction")
    print("=" * 50)

    # Initialize STAR model
    print("1. Initializing STAR model...")
    star_model = STARModel()

    # Load dataset
    print("\n2. Loading dataset...")
    texts, author_ids, unique_authors, author_to_id = load_stylometric_dataset()

    # Extract embeddings
    print("\n3. Extracting embeddings...")
    embeddings = star_model.extract_embeddings(texts, batch_size=32)
    
    # Save embeddings and metadata
    print("\n4. Saving embeddings and metadata...")
    embeddings_path, metadata_path = save_embeddings_and_metadata(
        embeddings, author_ids, unique_authors, author_to_id
    )
    
    print("\n" + "=" * 50)
    print("EMBEDDING EXTRACTION COMPLETE!")
    print("=" * 50)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Files saved in: /kaggle/working/")
    print("- star_embeddings.npy")
    print("- metadata.pkl")
    print("- metadata.json")
    
    # Memory cleanup
    del star_model
    torch.cuda.empty_cache()
    
    return embeddings_path, metadata_path

# Run the extraction
if __name__ == "__main__":
    embeddings_path, metadata_path = main_extraction()