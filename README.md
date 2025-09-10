# STAR Model Siamese Fine-tuning Implementation

## 📋 Overview

This project implements Siamese fine-tuning on top of the frozen STAR (Style Transformer for Authorship Representations) model for authorship verification tasks. The approach follows the methodology demonstrated by the STAR authors, where only task-specific layers are trained while keeping the pre-trained STAR embeddings frozen.

## 🏗️ Architecture Overview

```
Text Input → Frozen STAR Model → Style Embeddings (1024D) → Siamese Network → Similarity Score (0-1)
```

### Key Components:
1. **Frozen STAR Model**: Extracts 1024-dimensional style embeddings
2. **Siamese Network**: Learns pairwise similarity from concatenated embeddings
3. **Training Strategy**: Binary classification (same author vs different author)

## 🔧 Code Structure

### 1. STARModel Class
```python
class STARModel:
    def __init__(self, model_name='AIDA-UPM/star'):
        # Loads pre-trained STAR model
        # FREEZES all parameters (requires_grad=False)
        # Uses RoBERTa-large tokenizer
```

**Key Features:**
- **Parameter Freezing**: All STAR parameters are frozen to prevent updates
- **GPU Support**: Automatically uses CUDA if available
- **Batch Processing**: Efficient embedding extraction with configurable batch size

### 2. SiameseNetwork Class (Core Architecture)

```python
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dims=[1024, 512, 256, 128]):
        # Input: 2 * 1024 = 2048 (concatenated embeddings)
        # Architecture: 2048 → 1024 → 512 → 256 → 128 → 1
        # Output: Similarity score (0-1)
```

#### **Detailed Architecture Breakdown:**

```python
# Layer 1: Input → 1024
nn.Linear(2048, 1024)      # Concatenated embeddings
nn.BatchNorm1d(1024)       # Normalization
nn.ReLU()                  # Activation
nn.Dropout(0.4)            # Regularization

# Layer 2: 1024 → 512
nn.Linear(1024, 512)
nn.BatchNorm1d(512)
nn.ReLU()
nn.Dropout(0.4)

# Layer 3: 512 → 256
nn.Linear(512, 256)
nn.BatchNorm1d(256)
nn.ReLU()
nn.Dropout(0.3)

# Layer 4: 256 → 128
nn.Linear(256, 128)
nn.BatchNorm1d(128)
nn.ReLU()
nn.Dropout(0.3)

# Output Layer: 128 → 1
nn.Linear(128, 1)
nn.Sigmoid()               # Final similarity score
```

#### **Forward Pass Process:**
1. **Input**: Two 1024D embeddings (embedding1, embedding2)
2. **Concatenation**: `torch.cat([embedding1, embedding2], dim=1)` → 2048D
3. **Feature Learning**: Pass through 4-layer feed-forward network
4. **Output**: Single similarity score between 0-1

### 3. SiameseDataset Class

```python
class SiameseDataset(Dataset):
    def __init__(self, embeddings, author_ids, pairs_per_class=1000):
        # Generates training pairs for Siamese learning
        # Positive pairs: Same author (label=1)
        # Negative pairs: Different authors (label=0)
```

#### **Pair Generation Strategy:**
- **Balanced Sampling**: Ensures equal representation from all authors
- **Positive Pairs**: Randomly sample 2 texts from same author
- **Negative Pairs**: Sample 1 text from each of 2 different authors
- **Diversity**: Systematic author combination for better coverage

### 4. SiameseTrainer Class

```python
class SiameseTrainer:
    def __init__(self, siamese_model, device=None, learning_rate=0.0015):
        # Optimizer: AdamW with weight decay
        # Loss: Binary Cross-Entropy
        # Scheduler: ReduceLROnPlateau
```

#### **Training Configuration:**
- **Optimizer**: AdamW (lr=0.0015, weight_decay=5e-4)
- **Loss Function**: BCELoss for binary classification
- **Scheduler**: Reduces LR when validation accuracy plateaus
- **Early Stopping**: Stops after 5 epochs without improvement

## 🎯 How the Siamese Network Works

### 1. **Input Processing**
```python
# Two texts from the same or different authors
text1 = "This is a sample text by Author A"
text2 = "Another text by Author A"  # Same author = positive pair

# Extract embeddings using frozen STAR
embedding1 = star_model.extract_embeddings([text1])  # Shape: (1, 1024)
embedding2 = star_model.extract_embeddings([text2])  # Shape: (1, 1024)
```

### 2. **Siamese Processing**
```python
# Concatenate embeddings
combined = torch.cat([embedding1, embedding2], dim=1)  # Shape: (1, 2048)

# Pass through Siamese network
similarity_score = siamese_network(embedding1, embedding2)  # Shape: (1, 1)
# Output: 0.85 (high similarity for same author)
```

### 3. **Training Process**
```python
# For each training batch:
for embedding1, embedding2, labels in train_loader:
    # Forward pass
    predictions = siamese_network(embedding1, embedding2)
    
    # Compute loss
    loss = criterion(predictions, labels)  # labels: 1 for same author, 0 for different
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

## 🔧 How to Modify for Better Accuracy

### 1. **Architecture Modifications**

#### **A. Change Network Depth**
```python
# Current: 4 layers
hidden_dims=[1024, 512, 256, 128]

# Deeper network (more capacity)
hidden_dims=[1024, 768, 512, 384, 256, 128]

# Wider network (more parameters)
hidden_dims=[1536, 1024, 768, 512, 256]
```

#### **B. Modify Regularization**
```python
# Current dropout rates
nn.Dropout(0.4 if i < len(hidden_dims) - 2 else 0.3)

# Stronger regularization (if overfitting)
nn.Dropout(0.5 if i < len(hidden_dims) - 2 else 0.4)

# Weaker regularization (if underfitting)
nn.Dropout(0.2 if i < len(hidden_dims) - 2 else 0.1)
```

#### **C. Add Skip Connections**
```python
class SiameseNetwork(nn.Module):
    def forward(self, embedding1, embedding2):
        combined = torch.cat([embedding1, embedding2], dim=1)
        
        # Add skip connection
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Skip connection
        x = x + self.fc2(x)  # Residual connection
        
        return self.fc3(x)
```

### 2. **Training Modifications**

#### **A. Adjust Learning Rate**
```python
# Current: 0.0015
trainer = SiameseTrainer(siamese_net, learning_rate=0.002)  # Higher LR

# Or use learning rate scheduling
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
```

#### **B. Change Batch Size**
```python
# Current: 64
train_loader = DataLoader(train_dataset, batch_size=32)  # Smaller batches

# Or larger batches
train_loader = DataLoader(train_dataset, batch_size=128)  # Larger batches
```

#### **C. Modify Loss Function**
```python
# Current: BCELoss
self.criterion = nn.BCELoss()

# Focal Loss for hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
```

### 3. **Data Modifications**

#### **A. Increase Training Pairs**
```python
# Current: 1500 pairs per class
train_dataset = SiameseDataset(train_embeddings, train_author_ids, pairs_per_class=2000)

# Or use data augmentation
def augment_embeddings(embeddings, noise_factor=0.01):
    noise = torch.randn_like(embeddings) * noise_factor
    return embeddings + noise
```

#### **B. Improve Pair Generation**
```python
# Add hard negative mining
def generate_hard_negatives(self, embeddings, author_ids):
    # Find most similar embeddings from different authors
    # Use cosine similarity to find challenging negative pairs
    pass
```

### 4. **Advanced Techniques**

#### **A. Multi-Task Learning**
```python
class MultiTaskSiamese(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.shared_layers = nn.Sequential(...)
        self.similarity_head = nn.Linear(128, 1)
        self.author_classifier = nn.Linear(128, num_authors)
    
    def forward(self, embedding1, embedding2):
        combined = torch.cat([embedding1, embedding2], dim=1)
        features = self.shared_layers(combined)
        
        similarity = torch.sigmoid(self.similarity_head(features))
        author_pred = self.author_classifier(features)
        
        return similarity, author_pred
```

#### **B. Attention Mechanism**
```python
class AttentionSiamese(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.fc_layers = nn.Sequential(...)
    
    def forward(self, embedding1, embedding2):
        # Apply attention between embeddings
        attended, _ = self.attention(embedding1, embedding2, embedding2)
        combined = torch.cat([embedding1, attended], dim=1)
        return self.fc_layers(combined)
```

## 📊 Performance Monitoring

### Key Metrics to Track:
1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease without overfitting
3. **Validation Accuracy**: Target > 0.70
4. **Learning Rate**: Monitor scheduler effectiveness
5. **Gradient Norms**: Check for vanishing/exploding gradients

### Signs of Issues:
- **Overfitting**: Training loss ↓, Validation loss ↑
- **Underfitting**: Both losses plateau high
- **Learning Rate Too High**: Loss oscillates wildly
- **Learning Rate Too Low**: Very slow convergence

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch transformers scikit-learn pandas tqdm

# Run training
python Fine.py
```

## 📈 Expected Results

- **Baseline (Logistic Regression)**: ~0.47
- **Current Siamese Network**: ~0.58-0.65
- **Target with Optimizations**: 0.70-0.80+

## 🔍 Debugging Tips

1. **Check Data Balance**: Ensure equal positive/negative pairs
2. **Monitor Gradients**: Use `torch.nn.utils.clip_grad_norm_()`
3. **Visualize Embeddings**: Use t-SNE to check clustering
4. **Validate Architecture**: Test with simple synthetic data first
5. **Profile Performance**: Use `torch.profiler` for bottlenecks

## 📚 Key Files

- `Fine.py`: Main implementation with all classes
- `stylometric_dataset.csv`: Training data (1000 texts, 10 authors)
- `README.md`: This documentation

## 🎯 Next Steps for Better Accuracy

1. **Experiment with Architecture**: Try different layer sizes and depths
2. **Hyperparameter Tuning**: Use grid search or Bayesian optimization
3. **Data Augmentation**: Add noise or transformations to embeddings
4. **Ensemble Methods**: Train multiple models and average predictions
5. **Advanced Regularization**: Add L1/L2 penalties or dropout scheduling

This implementation provides a solid foundation for authorship verification with room for significant improvements through the modifications outlined above.
