"""
STAR Model Author Prediction - Cell 3 (Testing Cell)
This cell loads a trained model and predicts authors from text input.
"""

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# Add safe globals for LabelEncoder și numpy reconstruct
torch.serialization.add_safe_globals([LabelEncoder, np.core.multiarray._reconstruct])

# ==== LOAD CHECKPOINT ====
# Define the checkpoint path - modify this path as needed
checkpoint_path = '/kaggle/working/advanced_author_classifier.pth'

try:
    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
        weights_only=False  # allow full object loading
    )
except FileNotFoundError:
    print("❌ Checkpoint file not found!")
    exit()

# Extract components
model_state_dict = checkpoint['model_state_dict']
label_encoder = checkpoint['label_encoder']
num_classes = checkpoint['num_classes']
embedding_dim = checkpoint['embedding_dim']
model_config = checkpoint['model_config']

# Load metadata to get actual author names
import pickle
try:
    with open('/kaggle/working/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    unique_authors = metadata['unique_authors']
    author_to_id = metadata['author_to_id']
    id_to_author = {v: k for k, v in author_to_id.items()}
except FileNotFoundError:
    unique_authors = list(range(num_classes))
    id_to_author = {i: f"Author_{i}" for i in range(num_classes)}

# ==== RECREATE ADVANCED CLASSIFIER MODEL ====
# Make sure AdvancedClassifierNetwork is defined (from previous cell)
model = AdvancedClassifierNetwork(
    embedding_dim=embedding_dim,
    num_classes=num_classes,
    hidden_dims=model_config['hidden_dims'],
    num_transformer_layers=model_config['num_transformer_layers'],
    num_heads=model_config['num_heads'],
    dropout=model_config['dropout']
)

model.load_state_dict(model_state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# ==== STAR MODEL WRAPPER ====
# Make sure STARModel is defined (from previous cell)
star_model = STARModel()

# ==== PREDICTION FOR SINGLE EMBEDDING ====
def predict_author(embedding, model, label_encoder, device, id_to_author):
    model.eval()
    with torch.no_grad():
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        embedding = embedding.to(device).unsqueeze(0)
        outputs = model(embedding)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities.max().item()

        # Convert class index to author name using the mapping
        author_name = id_to_author[predicted_class]

        return author_name, confidence

# ==== PREDICTION FROM RAW TEXT ====
def predict_author_from_text(text, star_model, classifier_model, label_encoder, device, id_to_author):
    """
    Predict author given a raw text string
    """
    embedding = star_model.extract_embeddings([text], batch_size=1)  # shape (1, embedding_dim)
    author_name, confidence = predict_author(embedding[0], classifier_model, label_encoder, device, id_to_author)
    return author_name, confidence

# ==== BATCH PREDICTION ====
def predict_authors_from_texts(texts, star_model, classifier_model, label_encoder, device, id_to_author, batch_size=16):
    """
    Predict authors for multiple texts at once
    """
    embeddings = star_model.extract_embeddings(texts, batch_size=batch_size)
    results = [predict_author(e, classifier_model, label_encoder, device, id_to_author) for e in embeddings]
    return results

# ==== ENHANCED DISPLAY FUNCTIONS ====
def display_prediction(text, author, confidence, text_preview_length=100):
    """Display a single prediction in a nice format"""
    print("=" * 80)
    print(f"📝 TEXT: {text[:text_preview_length]}{'...' if len(text) > text_preview_length else ''}")
    print(f"👤 PREDICTED AUTHOR: {author}")
    print(f"🎯 CONFIDENCE: {confidence:.1%}")
    print("=" * 80)

def display_batch_predictions(texts, predictions):
    """Display multiple predictions in a nice format"""
    print("\n🔍 BATCH PREDICTION RESULTS:")
    print("=" * 100)
    
    for i, (text, (author, conf)) in enumerate(zip(texts, predictions), 1):
        print(f"\n{i}. TEXT: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"   AUTHOR: {author}")
        print(f"   CONFIDENCE: {conf:.1%}")
        print("-" * 100)

# ==== EXAMPLE BATCH ====

texts = [
    "Fran.... Kerri Rigsby from Associates will be contacting you regarding the Bridge Loan.  She needs a letter on file verifying that I will have these funds.  We still need to decide where we are going to put this money. For right now, we should plan on me signing the promisary note by fax, and then have a cheque sent up to me in Canada so that I can consolidate with the rest of the funds required for closing.  Thanks. BT",
    "I can't just sit here and write e-mails all day. I'm a busy man. I did not mean to offend you. I had smoozing to do and didn't want to neglect our customers."
]

predictions = predict_authors_from_texts(texts, star_model, model, label_encoder, device, id_to_author)
display_batch_predictions(texts, predictions)