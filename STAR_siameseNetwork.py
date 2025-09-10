"""
STAR Model Siamese Fine-tuning Implementation

This script implements Siamese fine-tuning on top of the frozen STAR (Style Transformer 
for Authorship Representations) model, following the approach demonstrated by the STAR authors.

Key Features:
- Freezes all STAR model parameters during training
- Trains only a task-specific Siamese network on pairwise embeddings
- Uses a small feed-forward network for similarity scoring
- Implements standard PyTorch training loop with validation metrics
- Limited to maximum 5 epochs as requested
- Outputs validation metrics after each epoch

Architecture:
1. Frozen STAR model extracts style embeddings
2. Siamese network takes pairs of embeddings as input
3. Feed-forward network outputs similarity scores (0-1)
4. Binary cross-entropy loss for same/different author classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
    
class STARModel:
    """STAR Model wrapper for easy testing and evaluation"""

    def __init__(self, model_name='AIDA-UPM/star'):
        """
        Initialize STAR model

        Args:
            model_name (str): Hugging Face model name
        """
        print(f"Loading STAR model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Freeze all STAR model parameters for fine-tuning
        self._freeze_star_parameters()
        print(f"Model loaded on device: {self.device}")
        print("STAR model parameters frozen for fine-tuning")
    
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

        print(f"Extracting embeddings for {len(texts)} texts...")
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
    Enhanced Siamese Network for fine-tuning on top of frozen STAR embeddings.
    Takes pairwise embeddings and outputs similarity scores with improved architecture.
    """
    
    def __init__(self, embedding_dim=1024, hidden_dims=[1024, 512, 256, 128]):
        """
        Initialize enhanced Siamese network
        
        Args:
            embedding_dim (int): Dimension of STAR embeddings (default: 1024)
            hidden_dims (list): List of hidden dimensions for deeper network
        """
        super(SiameseNetwork, self).__init__()
        
        # Enhanced feed-forward network with more layers and better regularization
        layers = []
        input_dim = 2 * embedding_dim  # Concatenated embeddings
        
        # Build deeper network with batch normalization and stronger regularization
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4 if i < len(hidden_dims) - 2 else 0.3)  # Stronger dropout to prevent overfitting
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with Xavier initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, embedding1, embedding2):
        """
        Forward pass through enhanced Siamese network
        
        Args:
            embedding1 (torch.Tensor): First embedding [batch_size, embedding_dim]
            embedding2 (torch.Tensor): Second embedding [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Similarity scores [batch_size, 1]
        """
        # Concatenate the embeddings
        combined = torch.cat([embedding1, embedding2], dim=1)
        
        # Pass through enhanced network
        output = self.network(combined)
        
        return output

class SiameseDataset(Dataset):
    """
    Dataset for Siamese network training.
    Generates pairs of embeddings with labels (same author = 1, different author = 0)
    """
    
    def __init__(self, embeddings, author_ids, pairs_per_class=1000):
        """
        Initialize Siamese dataset
        
        Args:
            embeddings (np.ndarray): Style embeddings [n_samples, embedding_dim]
            author_ids (list): Author IDs for each sample
            pairs_per_class (int): Number of positive and negative pairs to generate
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.author_ids = author_ids
        
        # Generate training pairs
        self.pairs, self.labels = self._generate_pairs(pairs_per_class)
        
    def _generate_pairs(self, pairs_per_class):
        """
        Generate enhanced training pairs with better balance and diversity
        
        Args:
            pairs_per_class (int): Number of pairs per class to generate
            
        Returns:
            tuple: (pairs, labels) where pairs is list of (idx1, idx2) and labels is list of 0/1
        """
        pairs = []
        labels = []
        
        # Group samples by author
        author_to_indices = {}
        for idx, author_id in enumerate(self.author_ids):
            if author_id not in author_to_indices:
                author_to_indices[author_id] = []
            author_to_indices[author_id].append(idx)
        
        authors = list(author_to_indices.keys())
        
        # Generate positive pairs (same author) with balanced sampling
        print(f"Generating {pairs_per_class} balanced positive pairs...")
        positive_count = 0
        attempts = 0
        max_attempts = pairs_per_class * 3  # Prevent infinite loops
        
        while positive_count < pairs_per_class and attempts < max_attempts:
            # Ensure balanced sampling across all authors
            author_id = authors[positive_count % len(authors)]
            if len(author_to_indices[author_id]) >= 2:
                idx1, idx2 = np.random.choice(author_to_indices[author_id], 2, replace=False)
                pairs.append((idx1, idx2))
                labels.append(1)  # Same author
                positive_count += 1
            attempts += 1
        
        # Generate negative pairs (different authors) with balanced sampling
        print(f"Generating {pairs_per_class} balanced negative pairs...")
        negative_count = 0
        attempts = 0
        
        while negative_count < pairs_per_class and attempts < max_attempts:
            # Ensure diverse negative pairs across different author combinations
            author1_idx = negative_count % len(authors)
            author2_idx = (negative_count + len(authors)//2) % len(authors)
            
            if author1_idx != author2_idx:
                author1 = authors[author1_idx]
                author2 = authors[author2_idx]
                idx1 = np.random.choice(author_to_indices[author1])
                idx2 = np.random.choice(author_to_indices[author2])
                pairs.append((idx1, idx2))
                labels.append(0)  # Different authors
                negative_count += 1
            attempts += 1
        
        print(f"Generated {len([l for l in labels if l == 1])} positive and {len([l for l in labels if l == 0])} negative pairs")
        return pairs, labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        embedding1 = self.embeddings[idx1]
        embedding2 = self.embeddings[idx2]
        label = torch.FloatTensor([self.labels[idx]])
        
        return embedding1, embedding2, label

class SiameseTrainer:
    """
    Trainer class for Siamese fine-tuning on frozen STAR embeddings.
    Implements standard PyTorch training loop with validation metrics.
    """
    
    def __init__(self, siamese_model, device=None, learning_rate=0.001):
        """
        Initialize enhanced Siamese trainer
        
        Args:
            siamese_model (SiameseNetwork): The Siamese network to train
            device (torch.device): Device to use for training
            learning_rate (float): Initial learning rate
        """
        self.siamese_model = siamese_model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.siamese_model.to(self.device)
        
        # Enhanced loss function with label smoothing for better generalization
        self.criterion = nn.BCELoss()
        
        # Better optimizer with stronger regularization
        self.optimizer = optim.AdamW(
            self.siamese_model.parameters(), 
            lr=learning_rate, 
            weight_decay=5e-4,  # Stronger weight decay to prevent overfitting
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # Monitor validation accuracy 
            factor=0.6,  # Less aggressive reduction
            patience=2,  # Reduce patience for faster adaptation
            min_lr=1e-6
        )
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss for the epoch
        """
        self.siamese_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for embedding1, embedding2, labels in tqdm(train_loader, desc="Training"):
            # Move data to device
            embedding1 = embedding1.to(self.device)
            embedding2 = embedding2.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.siamese_model(embedding1, embedding2)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.siamese_model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for embedding1, embedding2, labels in tqdm(val_loader, desc="Validating"):
                # Move data to device
                embedding1 = embedding1.to(self.device)
                embedding2 = embedding2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.siamese_model(embedding1, embedding2)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Compute accuracy (threshold at 0.5)
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=12):
        """
        Enhanced training loop with validation after each epoch and learning rate scheduling
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Maximum number of epochs (default: 12)
            
        Returns:
            dict: Training history with losses and accuracies
        """
        print(f"Starting enhanced Siamese fine-tuning for {num_epochs} epochs...")
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.siamese_model.parameters() if p.requires_grad):,}")
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        max_patience = 5  # Early stopping patience
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            
            # Update learning rate scheduler
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_accuracy)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Manual verbose output for learning rate changes
            if new_lr != old_lr:
                print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            
            # Save metrics
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['val_accuracies'].append(val_accuracy)
            history['learning_rates'].append(current_lr)
            
            # Print epoch results
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Save best model and early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                print(f"🎯 New best validation accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epoch(s)")
                
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"{'='*60}")
        return history

class STAREvaluator:
    """Evaluator for STAR model performance"""

    def __init__(self, star_model):
        self.star_model = star_model

    def evaluate_classification(self, embeddings, labels, test_size=0.2):
        """
        Evaluate classification performance using Logistic Regression

        Args:
            embeddings (np.ndarray): Style embeddings
            labels (list): Author labels
            test_size (float): Test set proportion

        Returns:
            dict: Classification evaluation metrics
        """
        print("Evaluating classification performance with Logistic Regression...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Train classifier
        classifier = LogisticRegression(
            random_state=42, max_iter=1000, multi_class='multinomial', solver='lbfgs'
        )
        classifier.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'y_true': y_test,
            'y_pred': y_pred
        }


def load_stylometric_dataset(csv_path='star_test_dataset.csv'):
    """
    Load the stylometric dataset from CSV file

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        tuple: (texts, authors) - Lists of texts and corresponding authors
    """
    print(f"Loading stylometric dataset from {csv_path}...")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract texts and authors
    texts = df['text'].tolist()
    authors = df['author'].tolist()

    # Convert author names to numeric IDs for easier processing
    unique_authors = list(set(authors))
    author_to_id = {author: idx for idx, author in enumerate(unique_authors)}
    author_ids = [author_to_id[author] for author in authors]

    print(f"Loaded {len(texts)} texts from {len(unique_authors)} authors")
    print(f"Authors: {unique_authors}")

    return texts, author_ids, unique_authors

def main():
    """Main function for STAR model testing and Siamese fine-tuning"""
    print("=" * 70)
    print("STAR (Style Transformer for Authorship Representations)")
    print("Siamese Fine-tuning Implementation")
    print("=" * 70)

    # Initialize STAR model with frozen parameters
    print("\n1. Initializing STAR model...")
    star_model = STARModel()

    # Load stylometric dataset
    print("\n2. Loading stylometric dataset...")
    texts, author_ids, unique_authors = load_stylometric_dataset()
    print(f"Dataset: {len(texts)} texts from {len(unique_authors)} authors")

    # Extract embeddings using frozen STAR model
    print("\n3. Extracting style embeddings...")
    embeddings = star_model.extract_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Baseline evaluation (optional)
    print("\n4. Baseline evaluation with Logistic Regression...")
    evaluator = STAREvaluator(star_model)
    baseline_results = evaluator.evaluate_classification(embeddings, author_ids)
    print(f"Baseline Accuracy: {baseline_results['accuracy']:.3f}")

    # Prepare data for Siamese fine-tuning
    print("\n5. Preparing Siamese training data...")
    
    # Create training and validation datasets
    # Split embeddings and author_ids for train/val
    train_embeddings, val_embeddings, train_author_ids, val_author_ids = train_test_split(
        embeddings, author_ids, test_size=0.2, random_state=42, stratify=author_ids
    )
    
    # Create enhanced Siamese datasets with more training pairs
    train_dataset = SiameseDataset(train_embeddings, train_author_ids, pairs_per_class=1500)
    val_dataset = SiameseDataset(val_embeddings, val_author_ids, pairs_per_class=400)
    
    # Create data loaders with better batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Training pairs: {len(train_dataset)}")
    print(f"Validation pairs: {len(val_dataset)}")

    # Initialize enhanced Siamese network
    print("\n6. Initializing enhanced Siamese network...")
    embedding_dim = embeddings.shape[1]  # Should be 1024 for STAR
    siamese_net = SiameseNetwork(
        embedding_dim=embedding_dim, 
        hidden_dims=[1024, 512, 256, 128]  # Deeper architecture
    )
    
    # Initialize trainer with optimized learning rate
    trainer = SiameseTrainer(siamese_net, device=star_model.device, learning_rate=0.0015)
    
    # Perform enhanced Siamese fine-tuning (up to 12 epochs with early stopping)
    print("\n7. Starting enhanced Siamese fine-tuning...")
    print("Training deeper task-specific network on top of frozen STAR embeddings...")
    print("Features: Early stopping, learning rate scheduling, batch normalization")
    training_history = trainer.train(train_loader, val_loader, num_epochs=12)
    
    # Print final results
    print("\n8. Final Results:")
    print("-" * 50)
    print(f"Baseline (Logistic Regression) Accuracy: {baseline_results['accuracy']:.3f}")
    print(f"Final Siamese Validation Accuracy: {training_history['val_accuracies'][-1]:.3f}")
    print(f"Best Siamese Validation Accuracy: {max(training_history['val_accuracies']):.3f}")
    
    # Print training progression
    print("\nTraining Progression:")
    for epoch, (train_loss, val_loss, val_acc) in enumerate(zip(
        training_history['train_losses'], 
        training_history['val_losses'], 
        training_history['val_accuracies']
    )):
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    print("\n" + "=" * 70)
    print("Siamese fine-tuning completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()