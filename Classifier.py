"""
STAR Multi-Class Classification - Advanced Author Attribution
Direct classification approach: STAR embedding -> Author (no pairwise comparison needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import warnings
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')

# ==== Advanced Multi-Class Classifier Definitions ====

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like attention"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with residual connections"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feedforward"""
    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4
            
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual connection
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class AdvancedClassifierNetwork(nn.Module):
    """Advanced Multi-Class Classifier with Transformer attention and multiple techniques"""
    
    def __init__(self, embedding_dim=1024, num_classes=100, hidden_dims=[768, 512, 256], 
                 num_transformer_layers=3, num_heads=12, dropout=0.15, use_mixup=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.use_mixup = use_mixup
        
        # Input normalization and projection
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, hidden_dims[0])
        
        # Positional encoding for sequence modeling
        self.pos_encoding = PositionalEncoding(hidden_dims[0])
        
        # Transformer layers for self-attention
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dims[0], num_heads, hidden_dims[0] * 2, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Feature extraction layers
        feature_layers = []
        input_dim = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            feature_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * (1 + i * 0.1))  # Increasing dropout in deeper layers
            ])
            input_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Multi-scale feature fusion
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 4),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1] // 4, 1)
        )
        
        # Final classification layers
        final_dim = hidden_dims[-1] * 2  # Concatenation of features and attention-weighted features
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.LayerNorm(final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, final_dim // 4),
            nn.LayerNorm(final_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(final_dim // 4, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform and specific strategies"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def mixup_data(self, x, y, alpha=0.4):
        """Mixup data augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

    def forward(self, embeddings, labels=None):
        # Input normalization and projection
        x = self.input_norm(embeddings)
        x = self.input_projection(x)
        
        # Add batch dimension for transformer if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Apply positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Remove sequence dimension and apply feature extraction
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        features = self.feature_extractor(x)
        
        # Simplified feature aggregation - no multi-scale pooling needed for dense features
        # Features is already (batch_size, feature_dim) after feature_extractor
        
        # Apply attention-based feature weighting
        attention_weights = F.softmax(self.attention_pool(features), dim=1)
        attention_features = features * attention_weights
        
        # Combine original features with attention-weighted features
        combined_features = torch.cat([features, attention_features], dim=1)
        
        # Final classification
        logits = self.classifier(combined_features)
        
        return logits


class ClassificationDataset(Dataset):
    """Dataset for direct multi-class classification"""
    
    def __init__(self, embeddings, author_ids, label_encoder=None):
        self.embeddings = torch.FloatTensor(embeddings)
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = torch.LongTensor(self.label_encoder.fit_transform(author_ids))
        else:
            self.label_encoder = label_encoder
            self.labels = torch.LongTensor(label_encoder.transform(author_ids))
        
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Dataset created with {len(self.embeddings)} samples across {self.num_classes} classes")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class AdvancedClassifierTrainer:
    """Advanced trainer with multiple optimization techniques"""
    
    def __init__(self, model, device=None, lr=2e-4, weight_decay=1e-4, 
                 use_focal_loss=True, focal_alpha=1, focal_gamma=2):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with different learning rates for different parts
        param_groups = [
            {'params': self.model.transformer_layers.parameters(), 'lr': lr * 0.5},
            {'params': self.model.feature_extractor.parameters(), 'lr': lr},
            {'params': self.model.classifier.parameters(), 'lr': lr * 1.5}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Advanced scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=[lr * 0.5, lr, lr * 1.5], 
            epochs=20, steps_per_epoch=100, pct_start=0.3
        )
        
        # For tracking
        self.best_val_acc = 0.0
        self.best_model_state = None

    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Mixup loss calculation"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_epoch(self, loader, use_mixup=True):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for embeddings, labels in tqdm(loader, desc="Training", leave=False):
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if use_mixup and self.model.use_mixup and self.model.training:
                mixed_embeddings, labels_a, labels_b, lam = self.model.mixup_data(embeddings, labels)
                outputs = self.model(mixed_embeddings)
                loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)
            else:
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (for non-mixup case)
            if not (use_mixup and self.model.use_mixup and self.model.training):
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            else:
                # For mixup, we'll calculate accuracy differently
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (lam * predicted.eq(labels_a).float() + 
                           (1 - lam) * predicted.eq(labels_b).float()).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        return total_loss / len(loader), accuracy

    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(loader, desc="Validation", leave=False):
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Save best model
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.best_model_state = self.model.state_dict().copy()
        
        return total_loss / len(loader), accuracy, f1

    def predict(self, embeddings):
        """Predict author for new embeddings"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.FloatTensor(embeddings)
            embeddings = embeddings.to(self.device)
            
            if len(embeddings.shape) == 1:
                embeddings = embeddings.unsqueeze(0)
            
            outputs = self.model(embeddings)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            return predicted.cpu().numpy(), probabilities.cpu().numpy()


# ==== MAIN TRAINING PIPELINE ====
print("Loading embeddings and metadata...")
embeddings = np.load('/kaggle/working/star_embeddings.npy')
with open('/kaggle/working/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
author_ids = metadata['author_ids']
print(f"Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

# Calculate unique authors and class distribution
unique_authors = np.unique(author_ids)
num_classes = len(unique_authors)
print(f"Number of unique authors: {num_classes}")

# Check class distribution
from collections import Counter
class_dist = Counter(author_ids)
print(f"Class distribution stats:")
print(f"  Min samples per class: {min(class_dist.values())}")
print(f"  Max samples per class: {max(class_dist.values())}")
print(f"  Mean samples per class: {np.mean(list(class_dist.values())):.2f}")

# Split data for classification (stratified)
train_emb, val_emb, train_ids, val_ids = train_test_split(
    embeddings, author_ids, test_size=0.2, stratify=author_ids, random_state=42
)

print(f"Training set: {train_emb.shape[0]} samples")
print(f"Validation set: {val_emb.shape[0]} samples")

# Create classification datasets
train_dataset = ClassificationDataset(train_emb, train_ids)
val_dataset = ClassificationDataset(val_emb, val_ids, label_encoder=train_dataset.label_encoder)

# Create data loaders with optimal batch size
batch_size = min(64, len(train_dataset) // 50)  # Dynamic batch size
print(f"Using batch size: {batch_size}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True,
    num_workers=0  # Set to 0 for compatibility
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    drop_last=False,
    num_workers=0
)

# Create the advanced classifier model
model = AdvancedClassifierNetwork(
    embedding_dim=embeddings.shape[1],
    num_classes=num_classes,
    hidden_dims=[768, 512, 256],
    num_transformer_layers=2,  # Reduced for faster training
    num_heads=12,
    dropout=0.15,
    use_mixup=True
)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Create the advanced trainer
trainer = AdvancedClassifierTrainer(
    model, 
    lr=2e-4, 
    weight_decay=1e-4,
    use_focal_loss=True,
    focal_alpha=1,
    focal_gamma=2
)

# Update scheduler with correct steps_per_epoch
trainer.scheduler = optim.lr_scheduler.OneCycleLR(
    trainer.optimizer, 
    max_lr=[2e-4 * 0.5, 2e-4, 2e-4 * 1.5], 
    epochs=20, 
    steps_per_epoch=len(train_loader), 
    pct_start=0.3
)

print(f"\nStarting advanced multi-class training...")
print(f"Device: {trainer.device}")
print("=" * 60)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 40)
    
    # Training
    train_loss, train_acc = trainer.train_epoch(train_loader, use_mixup=True)
    
    # Validation
    val_loss, val_acc, val_f1 = trainer.validate(val_loader)
    
    # Print metrics
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    print(f"Best Val Acc: {trainer.best_val_acc:.4f}")
    
    # Learning rate info
    current_lrs = [group['lr'] for group in trainer.optimizer.param_groups]
    print(f"Current LRs: {[f'{lr:.2e}' for lr in current_lrs]}")

print("\n" + "=" * 60)
print("Training completed!")
print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")

# Load best model for inference
if trainer.best_model_state is not None:
    model.load_state_dict(trainer.best_model_state)
    print("Loaded best model weights for inference.")

# Save the trained model and label encoder
print("Saving model and label encoder...")
torch.save({
    'model_state_dict': trainer.best_model_state or model.state_dict(),
    'label_encoder': train_dataset.label_encoder,
    'num_classes': num_classes,
    'embedding_dim': embeddings.shape[1],
    'model_config': {
        'hidden_dims': [768, 512, 256],
        'num_transformer_layers': 2,
        'num_heads': 12,
        'dropout': 0.15
    }
}, '/kaggle/working/advanced_author_classifier.pth')

print("Model saved successfully!")

# Example inference function
def predict_author(embedding, model, label_encoder, device):
    """
    Predict author for a single embedding
    
    Args:
        embedding: STAR embedding (numpy array or tensor)
        model: Trained model
        label_encoder: Fitted LabelEncoder
        device: Device to use for inference
    
    Returns:
        author_name, confidence_score
    """
    model.eval()
    with torch.no_grad():
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        
        embedding = embedding.to(device).unsqueeze(0)  # Add batch dimension
        
        outputs = model(embedding)
        probabilities = F.softmax(outputs, dim=1)
        
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities.max().item()
        
        author_name = label_encoder.inverse_transform([predicted_class])[0]
        
        return author_name, confidence

print("\nExample usage:")
print("predicted_author, confidence = predict_author(new_embedding, model, train_dataset.label_encoder, trainer.device)")
print("print(f'Predicted author: {predicted_author} (confidence: {confidence:.3f})')")

# Performance analysis on validation set
print("\nFinal performance analysis on validation set...")
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for embeddings, labels in val_loader:
        embeddings, labels = embeddings.to(trainer.device), labels.to(trainer.device)
        outputs = model(embeddings)
        probabilities = F.softmax(outputs, dim=1)
        
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

# Calculate detailed metrics
final_accuracy = accuracy_score(all_labels, all_preds)
final_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
final_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\nFinal Validation Results:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"F1-Score (Macro): {final_f1_macro:.4f}")
print(f"F1-Score (Weighted): {final_f1_weighted:.4f}")

# Top-K accuracy
all_probs = np.array(all_probs)
top_3_acc = np.mean([label in np.argsort(probs)[-3:] for label, probs in zip(all_labels, all_probs)])
top_5_acc = np.mean([label in np.argsort(probs)[-5:] for label, probs in zip(all_labels, all_probs)])

print(f"Top-3 Accuracy: {top_3_acc:.4f}")
print(f"Top-5 Accuracy: {top_5_acc:.4f}")

print("\n🎉 Advanced Multi-Class Classification completed successfully!")
print("✅ Advantages of this approach:")
print("   - Direct prediction: embedding → author (no dataset comparison needed)")
print("   - Fast inference: O(1) prediction time")
print("   - Advanced architecture with Transformer attention")
print("   - Multiple optimization techniques (Focal Loss, Mixup, etc.)")
print("   - Comprehensive evaluation metrics")
print("\n⚠️  Note: Cannot easily generalize to new authors not seen during training")
print("   For new author support, consider few-shot learning or retraining approaches.")