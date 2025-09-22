"""
STAR Model Ensemble Siamese Fine-tuning Implementation

This script implements an ensemble Siamese network approach on top of the frozen STAR model.

Key Features:
- Ensemble of multiple Siamese networks with different architectures
- Multiple similarity metrics (cosine, euclidean, element-wise operations)
- Attention mechanism for embedding fusion
- Contrastive loss for better similarity learning
- Hard negative mining for improved training
- Learnable ensemble weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import warnings
try:
    import faiss
except ImportError:
    print("Installing FAISS for Kaggle...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("Installing visualization packages...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn", "plotly", "umap-learn"])
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    import plotly.express as px
    import plotly.graph_objects as go

warnings.filterwarnings('ignore')

class STARModel:
    """STAR Model wrapper for easy testing and evaluation"""

    def __init__(self, model_name='AIDA-UPM/star'):
        """
        Initialize STAR model

        Args:
            model_name (str): Hugging Face model name
        """
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Freeze all STAR model parameters for fine-tuning
        self._freeze_star_parameters()

    def _freeze_star_parameters(self):
        """
        Freeze all STAR model parameters to prevent them from being updated during fine-tuning.
        Only the task-specific Siamese network will be trained.
        """
        for param in self.model.parameters():
            param.requires_grad = False

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

        for i in tqdm(range(0, len(texts), batch_size)):
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


class SiameseNetwork(nn.Module):
    """
    Base Siamese Network with multiple similarity metrics and attention mechanism
    """

    def __init__(self, embedding_dim=1024, hidden_dims=[512, 256, 128]):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embedding transformation layers
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism for embedding fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=16, 
            dropout=0.05,
            batch_first=True
        )
        
        # Multiple similarity computation methods
        similarity_features = 512 * 4 + 3  # concat + elementwise + abs_diff + cosine + euclidean + dot
        
        # Enhanced classification network
        layers = []
        input_dim = similarity_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def compute_similarities(self, emb1, emb2):
        """Compute multiple similarity metrics"""
        # Ensure embeddings are normalized
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        # 1. Concatenation
        concat = torch.cat([emb1_norm, emb2_norm], dim=1)
        
        # 2. Element-wise product (interaction)
        elementwise = emb1_norm * emb2_norm
        
        # 3. Absolute difference
        abs_diff = torch.abs(emb1_norm - emb2_norm)
        
        # 4. Cosine similarity (scalar)
        cosine_sim = F.cosine_similarity(emb1_norm, emb2_norm, dim=1, eps=1e-8).unsqueeze(1)
        
        # 5. Euclidean distance (scalar)
        euclidean_dist = torch.norm(emb1_norm - emb2_norm, p=2, dim=1, keepdim=True)
        
        # 6. Dot product (scalar)
        dot_product = (emb1_norm * emb2_norm).sum(dim=1, keepdim=True)
        
        # Combine all features
        features = torch.cat([
            concat, elementwise, abs_diff, 
            cosine_sim, euclidean_dist, dot_product
        ], dim=1)
        
        return features
    
    def forward(self, embedding1, embedding2):
        # Transform embeddings
        emb1_transformed = self.transform(embedding1)
        emb2_transformed = self.transform(embedding2)
        
        # Apply attention (self-attention on concatenated embeddings)
        combined = torch.cat([emb1_transformed.unsqueeze(1), emb2_transformed.unsqueeze(1)], dim=1)
        attended, _ = self.attention(combined, combined, combined)
        
        # Extract attended embeddings
        emb1_attended = attended[:, 0, :]
        emb2_attended = attended[:, 1, :]
        
        # Compute multiple similarities
        similarity_features = self.compute_similarities(emb1_attended, emb2_attended)
        
        # Classification
        output = self.classifier(similarity_features)
        
        return torch.sigmoid(output)

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks
    Better than BCE for similarity learning
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output, label):
        # Convert similarity to distance
        distance = 1 - output
        
        # Contrastive loss
        loss_contrastive = torch.mean(
            label * torch.pow(distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        
        return loss_contrastive

class SiameseDataset(Dataset):
    """
    Improved dataset with better pair generation and hard negative mining
    """
    
    def __init__(self, embeddings, author_ids, pairs_per_class=5000, hard_negative_ratio=0.3):
        self.embeddings = torch.FloatTensor(embeddings)
        self.author_ids = author_ids
        self.hard_negative_ratio = hard_negative_ratio
        
        # Generate improved training pairs
        self.pairs, self.labels = self._generate_improved_pairs(pairs_per_class)
    
    def _generate_improved_pairs(self, pairs_per_class):
        """Generate improved training pairs with hard negatives"""
        pairs = []
        labels = []
        
        # Group samples by author
        author_to_indices = {}
        for idx, author_id in enumerate(self.author_ids):
            if author_id not in author_to_indices:
                author_to_indices[author_id] = []
            author_to_indices[author_id].append(idx)
        
        authors = list(author_to_indices.keys())
        
        # Precompute embeddings for hard negative mining
        embeddings_np = self.embeddings.numpy()
        
        # Generate positive pairs
        positive_pairs = []
        for _ in range(pairs_per_class):
            author_id = np.random.choice(authors)
            if len(author_to_indices[author_id]) >= 2:
                idx1, idx2 = np.random.choice(author_to_indices[author_id], 2, replace=False)
                positive_pairs.append((idx1, idx2, 1))
        
        # Generate negative pairs (including hard negatives)
        negative_pairs = []
        hard_negative_count = int(pairs_per_class * self.hard_negative_ratio)
        easy_negative_count = pairs_per_class - hard_negative_count
        
        # Easy negatives (random different authors)
        for _ in range(easy_negative_count):
            author1, author2 = np.random.choice(authors, 2, replace=False)
            idx1 = np.random.choice(author_to_indices[author1])
            idx2 = np.random.choice(author_to_indices[author2])
            negative_pairs.append((idx1, idx2, 0))
        
        # Hard negatives using FAISS for efficient similarity search
        print("Building FAISS index for hard negative mining...")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        
        # Build FAISS index for fast similarity search
        embedding_dim = embeddings_normalized.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (cosine similarity for normalized vectors)
        index.add(embeddings_normalized.astype('float32'))
        
        # Mine hard negatives
        for _ in range(hard_negative_count):
            # Pick a random anchor
            anchor_idx = np.random.choice(len(self.embeddings))
            anchor_author = self.author_ids[anchor_idx]
            anchor_embedding = embeddings_normalized[anchor_idx:anchor_idx+1].astype('float32')
            
            # Find top-k most similar embeddings using FAISS
            k = min(50, len(self.embeddings))  # Look at top 50 most similar
            similarities, candidate_indices = index.search(anchor_embedding, k)
            
            # Find the most similar embedding from a different author
            best_idx = None
            for i, candidate_idx in enumerate(candidate_indices[0]):
                if candidate_idx != anchor_idx and self.author_ids[candidate_idx] != anchor_author:
                    best_idx = candidate_idx
                    break
            
            if best_idx is not None:
                negative_pairs.append((anchor_idx, best_idx, 0))
        
        # Combine all pairs
        all_pairs = positive_pairs + negative_pairs
        np.random.shuffle(all_pairs)
        
        pairs = [(p[0], p[1]) for p in all_pairs]
        labels = [p[2] for p in all_pairs]
        
        print(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")
        
        return pairs, labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        embedding1 = self.embeddings[idx1]
        embedding2 = self.embeddings[idx2]
        label = torch.FloatTensor([self.labels[idx]])
        
        return embedding1, embedding2, label




class EnsembleSiamese(nn.Module):
    """
    Ensemble of multiple Siamese networks for better performance
    """
    
    def __init__(self, embedding_dim=1024, num_models=3):
        super(EnsembleSiamese, self).__init__()
        
        self.models = nn.ModuleList([
            SiameseNetwork(embedding_dim, [512, 256, 128]),
            SiameseNetwork(embedding_dim, [256, 128, 64]),
            SiameseNetwork(embedding_dim, [384, 192, 96])
        ])
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, embedding1, embedding2):
        outputs = []
        for model in self.models:
            outputs.append(model(embedding1, embedding2))
        
        # Weighted average
        ensemble_output = torch.stack(outputs, dim=-1)
        weights = F.softmax(self.ensemble_weights, dim=0)
        final_output = (ensemble_output * weights).sum(dim=-1)
        
        return final_output

class EnsembleSiameseTrainer:
    """
    Enhanced trainer specifically designed for ensemble Siamese networks
    """
    
    def __init__(self, ensemble_model, device=None, learning_rate=0.0005):
        self.ensemble_model = ensemble_model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble_model.to(self.device)
        
        # Use contrastive loss
        self.criterion = ContrastiveLoss(margin=1.0)
        
        # Lower learning rate for ensemble training
        self.optimizer = optim.AdamW(
            self.ensemble_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler with longer cycle for ensemble
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=7,
            T_mult=2,
            eta_min=1e-7
        )
    
    def train_epoch(self, train_loader):
        self.ensemble_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        ensemble_weights_history = []
        
        for embedding1, embedding2, labels in tqdm(train_loader, desc=f"Training Epoch", leave=False):
            embedding1 = embedding1.to(self.device)
            embedding2 = embedding2.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.ensemble_model(embedding1, embedding2)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for ensemble stability
            torch.nn.utils.clip_grad_norm_(self.ensemble_model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Track ensemble weights
            with torch.no_grad():
                weights = F.softmax(self.ensemble_model.ensemble_weights, dim=0)
                ensemble_weights_history.append(weights.cpu().numpy())
        
        accuracy = correct / total
        avg_ensemble_weights = np.mean(ensemble_weights_history, axis=0)
        
        return total_loss / len(train_loader), accuracy, avg_ensemble_weights
    
    def validate(self, val_loader):
        self.ensemble_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            for embedding1, embedding2, labels in tqdm(val_loader, desc=f"Validation", leave=False):
                embedding1 = embedding1.to(self.device)
                embedding2 = embedding2.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.ensemble_model(embedding1, embedding2)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        
        accuracy = correct / total
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        
        # Calculate ensemble weights at validation
        with torch.no_grad():
            ensemble_weights = F.softmax(self.ensemble_model.ensemble_weights, dim=0).cpu().numpy()
        
        return total_loss / len(val_loader), accuracy, f1, precision, recall, ensemble_weights
    
    def train(self, train_loader, val_loader, num_epochs=5):
        print(f"Training ensemble for {num_epochs} epochs...")
        
        best_f1 = 0.0
        best_accuracy = 0.0
        patience_counter = 0
        max_patience = 10
        
        history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'val_f1_scores': [],
            'learning_rates': [],
            'ensemble_weights_history': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc, train_ensemble_weights = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall, val_ensemble_weights = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store history
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_acc)
            history['val_losses'].append(val_loss)
            history['val_accuracies'].append(val_acc)
            history['val_f1_scores'].append(val_f1)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            history['ensemble_weights_history'].append(val_ensemble_weights)
            
            # Print detailed progress for each epoch
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Ensemble Weights: [{', '.join([f'{w:.3f}' for w in val_ensemble_weights])}]")
            
            # Early stopping based on F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_accuracy = val_acc
                patience_counter = 0
                print(f"🎯 New best F1 score: {best_f1:.4f} (Accuracy: {best_accuracy:.4f})")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{max_patience}")
                
            if patience_counter >= max_patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        return {
            "best_f1": best_f1, 
            "best_accuracy": best_accuracy,
            "history": history,
            "final_ensemble_weights": val_ensemble_weights
        }

class EmbeddingVisualizer:
    """
    Visualize embeddings and clustering to see how well the model separates authors
    """
    
    def __init__(self, embeddings, author_ids, unique_authors):
        self.embeddings = embeddings
        self.author_ids = np.array(author_ids)
        self.unique_authors = unique_authors
        
    def plot_tsne_clustering(self, perplexity=30, random_state=42):
        """
        Create t-SNE visualization of embeddings colored by true authors
        """
        print("Computing t-SNE visualization...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, verbose=1)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create interactive plot with Plotly
        colors = px.colors.qualitative.Set3[:len(self.unique_authors)]
        
        fig = go.Figure()
        
        for i, author in enumerate(self.unique_authors):
            mask = self.author_ids == i
            fig.add_trace(go.Scatter(
                x=embeddings_2d[mask, 0],
                y=embeddings_2d[mask, 1],
                mode='markers',
                name=f'Author {author}',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                text=[f'Author: {author}' for _ in range(mask.sum())],
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='t-SNE Visualization of STAR Embeddings by Author',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            width=800,
            height=600,
            showlegend=True
        )
        
        fig.show()
        return embeddings_2d
    
    def analyze_clustering_performance(self, embeddings_2d=None):
        """
        Analyze how well different clustering algorithms separate the authors
        """
        if embeddings_2d is None:
            # Use PCA for faster dimensionality reduction
            pca = PCA(n_components=50, random_state=42)
            embeddings_reduced = pca.fit_transform(self.embeddings)
        else:
            embeddings_reduced = embeddings_2d
            
        n_authors = len(self.unique_authors)
        
        # Test different clustering algorithms
        clustering_results = {}
        
        print(f"Testing clustering algorithms for {n_authors} authors...")
        
        # K-Means clustering
        print("Testing K-Means...")
        kmeans = KMeans(n_clusters=n_authors, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings_reduced)
        clustering_results['K-Means'] = kmeans_labels
        
        # DBSCAN clustering - removed (not effective for authorship)
        # print("Testing DBSCAN...")
        # dbscan = DBSCAN(eps=0.5, min_samples=5)
        # dbscan_labels = dbscan.fit_predict(embeddings_reduced)
        # clustering_results['DBSCAN'] = dbscan_labels
        
        # Calculate clustering metrics
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
        
        print("\nClustering Performance Analysis:")
        print("=" * 50)
        
        for method, predicted_labels in clustering_results.items():
            if len(set(predicted_labels)) > 1:  # Valid clustering
                ari = adjusted_rand_score(self.author_ids, predicted_labels)
                nmi = normalized_mutual_info_score(self.author_ids, predicted_labels)
                silhouette = silhouette_score(embeddings_reduced, predicted_labels)
                
                print(f"{method}:")
                print(f"  - Adjusted Rand Index: {ari:.4f}")
                print(f"  - Normalized Mutual Info: {nmi:.4f}")
                print(f"  - Silhouette Score: {silhouette:.4f}")
                print(f"  - Number of clusters found: {len(set(predicted_labels))}")
                print()
            else:
                print(f"{method}: Failed to find meaningful clusters")
                print()
        
        return clustering_results
    
    def plot_clustering_comparison(self, embeddings_2d=None):
        """
        Compare true author labels vs predicted clusters visually
        """
        if embeddings_2d is None:
            print("Computing PCA for visualization...")
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(self.embeddings)
        
        # Get clustering results
        clustering_results = self.analyze_clustering_performance(embeddings_2d)
        
        # Create single plot - only K-Means clustering
        plt.figure(figsize=(8, 6))
        
        # Plot K-Means clustering results
        if 'K-Means' in clustering_results:
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=clustering_results['K-Means'], cmap='tab10', alpha=0.7)
            plt.title('K-Means Clustering Results')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.colorbar(scatter)
        
        
        plt.tight_layout()
        plt.show()
        
        return embeddings_2d, clustering_results
    
    # analyze_author_separability method removed
    # def analyze_author_separability(self):
    #     """Analyze how well separated each author is from others"""
    #     # Heatmap analysis removed per user request
    #     pass

def load_stylometric_dataset(csv_path='/kaggle/input/dataset2/emails_dataset_min70.csv'):
    """
    Load the stylometric dataset from CSV file
    """
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    authors = df['author'].tolist()
    
    unique_authors = list(set(authors))
    author_to_id = {author: idx for idx, author in enumerate(unique_authors)}
    author_ids = [author_to_id[author] for author in authors]

    return texts, author_ids, unique_authors


def train_ensemble_siamese(star_model, embeddings, author_ids):
    """
    Train ensemble Siamese network
    """
    print("Starting Ensemble Siamese training...")
    
    # Split data
    train_embeddings, val_embeddings, train_author_ids, val_author_ids = train_test_split(
        embeddings, author_ids, test_size=0.2, random_state=42, stratify=author_ids
    )
    
    # Create datasets
    train_dataset = SiameseDataset(
        train_embeddings, train_author_ids, 
        pairs_per_class=12000,
        hard_negative_ratio=0.35
    )
    
    val_dataset = SiameseDataset(
        val_embeddings, val_author_ids,
        pairs_per_class=5000,
        hard_negative_ratio=0.3
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0)
    
    # Create ensemble model
    embedding_dim = embeddings.shape[1]
    ensemble_model = EnsembleSiamese(embedding_dim=embedding_dim, num_models=3)
    
    # Train ensemble
    trainer = EnsembleSiameseTrainer(
        ensemble_model, 
        device=star_model.device,
        learning_rate=0.0005
    )
    
    results = trainer.train(train_loader, val_loader, num_epochs=3)
    
    print(f"Training complete - Best F1: {results['best_f1']:.4f}, Best Accuracy: {results['best_accuracy']:.4f}")
    
    return ensemble_model, results


def main():
    """Main function for STAR model Ensemble Siamese fine-tuning"""
    print("STAR Ensemble Siamese Network Training")
    print("=" * 50)

    # Initialize STAR model
    print("1. Initializing STAR model...")
    star_model = STARModel()

    # Load dataset
    print("2. Loading dataset...")
    texts, author_ids, unique_authors = load_stylometric_dataset()
    print(f"Loaded {len(texts)} texts from {len(unique_authors)} authors")

    # Extract embeddings
    print("3. Extracting embeddings...")
    embeddings = star_model.extract_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Train ensemble model
    print("4. Training ensemble model...")
    ensemble_model, results = train_ensemble_siamese(star_model, embeddings, author_ids)
    
    # Print results
    print("\nTraining completed successfully!")
    print(f"Best F1 Score: {results['best_f1']:.4f}")
    print(f"Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"Final Ensemble Weights: {results['final_ensemble_weights']}")
    
    # Create visualizations to analyze clustering performance
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS AND CLUSTERING ANALYSIS")
    print("="*60)
    
    visualizer = EmbeddingVisualizer(embeddings, author_ids, unique_authors)
    
    # 1. t-SNE visualization
    print("\n1. Creating t-SNE visualization...")
    embeddings_2d = visualizer.plot_tsne_clustering()
    
    # 2. Clustering comparison
    print("\n2. Comparing clustering algorithms...")
    embeddings_2d, clustering_results = visualizer.plot_clustering_comparison(embeddings_2d)
    
    # 3. Author separability analysis - removed
    # print("\n3. Analyzing author separability...")
    # distance_matrix = visualizer.analyze_author_separability()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    
    return star_model, ensemble_model, results, visualizer

if __name__ == "__main__":
    star_model, ensemble_model, results, visualizer = main()
