#!/usr/bin/env python3
"""
Script to train a custom spaCy model with URGENCY entity recognition
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from pathlib import Path

# Training data - format: (text, {"entities": [(start, end, label)]})
TRAINING_DATA = [
    # Urgent/Priority examples
    ("This is urgent, please respond immediately!", {"entities": [(8, 14, "URGENCY"), (33, 44, "URGENCY")]}),
    ("ASAP delivery needed for the project", {"entities": [(0, 4, "URGENCY")]}),
    ("Emergency meeting scheduled for today", {"entities": [(0, 9, "URGENCY")]}),
    ("Critical issue needs fixing now", {"entities": [(0, 8, "URGENCY"), (26, 29, "URGENCY")]}),
    ("Rush order for tomorrow deadline", {"entities": [(0, 4, "URGENCY"), (20, 28, "URGENCY")]}),
    ("High priority task must be completed soon", {"entities": [(5, 13, "URGENCY"), (36, 40, "URGENCY")]}),
    ("Please handle this quickly", {"entities": [(19, 26, "URGENCY")]}),
    ("Fast response required for this request", {"entities": [(0, 4, "URGENCY")]}),
    ("The contract expires today", {"entities": [(13, 20, "URGENCY")]}),
    ("Overdue payment needs attention now", {"entities": [(0, 7, "URGENCY"), (32, 35, "URGENCY")]}),
    
    # Normal text without urgency (important for training)
    ("John Smith works at Microsoft in Seattle", {"entities": [(0, 10, "PERSON"), (20, 29, "ORG"), (33, 40, "GPE")]}),
    ("The meeting is scheduled for next week", {"entities": []}),
    ("Please review the document when you have time", {"entities": []}),
    ("Sarah Johnson sent an email yesterday", {"entities": [(0, 13, "PERSON")]}),
    ("The office is located in New York", {"entities": [(25, 33, "GPE")]}),
    ("We received $10,000 for the project", {"entities": [(12, 19, "MONEY")]}),
    ("The presentation went well", {"entities": []}),
    ("Thank you for your cooperation", {"entities": []}),
    ("Google announced new features", {"entities": [(0, 6, "ORG")]}),
    ("The report will be ready next month", {"entities": []}),
    
    # Mixed examples with urgency and other entities
    ("Sarah needs urgent help with Microsoft project", {"entities": [(0, 5, "PERSON"), (12, 18, "URGENCY"), (29, 38, "ORG")]}),
    ("Emergency call from John at 3 PM today", {"entities": [(0, 9, "URGENCY"), (20, 24, "PERSON"), (28, 32, "TIME")]}),
    ("Critical deadline for Apple contract expires tomorrow", {"entities": [(0, 8, "URGENCY"), (13, 21, "URGENCY"), (26, 31, "ORG"), (41, 53, "URGENCY")]}),
    ("ASAP review needed for $50,000 budget", {"entities": [(0, 4, "URGENCY"), (23, 30, "MONEY")]}),
    ("Rush delivery to New York office now", {"entities": [(0, 4, "URGENCY"), (17, 25, "GPE"), (33, 36, "URGENCY")]}),
]

def create_blank_model():
    """Create a blank spaCy model with NER component"""
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    
    # Add the new label
    ner.add_label("URGENCY")
    
    return nlp

def update_existing_model():
    """Update existing spaCy model with new URGENCY label"""
    try:
        # Load existing model
        nlp = spacy.load("en_core_web_trf")
        print("Loaded existing en_core_web_trf model")
        
        # Get NER component
        ner = nlp.get_pipe("ner")
        
        # Add new label
        ner.add_label("URGENCY")
        print("Added URGENCY label to existing model")
        
        return nlp
        
    except OSError:
        print("Could not load en_core_web_trf, creating blank model")
        return create_blank_model()

def train_model(nlp, training_data, iterations=20):
    """Train the model with the training data"""
    
    # Get the NER component
    ner = nlp.get_pipe("ner")
    
    # Disable other pipeline components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    # Prepare training examples
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    
    # Start training
    with nlp.disable_pipes(*other_pipes):
        # Initialize the model with training examples
        nlp.initialize(lambda: examples)
        
        print(f"Starting training for {iterations} iterations...")
        
        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}")
            
            # Shuffle training data
            random.shuffle(examples)
            losses = {}
            
            # Create minibatches
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                # Update the model
                nlp.update(batch, drop=0.35, losses=losses)
            
            print(f"  Losses: {losses}")
    
    return nlp

def test_model(nlp):
    """Test the trained model with sample texts"""
    test_texts = [
        "This is urgent! Please respond ASAP.",
        "Emergency meeting with John Smith at Microsoft.",
        "Critical deadline tomorrow for the project.",
        "Regular meeting next week in New York.",
        "Rush delivery needed now!",
    ]
    
    print("\nTesting trained model:")
    print("=" * 50)
    
    for text in test_texts:
        doc = nlp(text)
        print(f"\nText: '{text}'")
        print("Entities found:")
        
        if doc.ents:
            for ent in doc.ents:
                print(f"  - '{ent.text}' -> {ent.label_} ({spacy.explain(ent.label_) if ent.label_ != 'URGENCY' else 'Urgency indicator'})")
        else:
            print("  - No entities found")

def save_model(nlp, output_dir="./custom_urgency_model"):
    """Save the trained model"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir()
    
    nlp.to_disk(output_path)
    print(f"Model saved to {output_path}")
    
    return output_path

def main():
    """Main training function"""
    print("Training Custom spaCy Model with URGENCY Entity Recognition")
    print("=" * 60)
    
    # Step 1: Load or create model
    print("\nStep 1: Loading base model...")
    nlp = update_existing_model()
    
    # Step 2: Train the model
    print("\nStep 2: Training model...")
    nlp = train_model(nlp, TRAINING_DATA, iterations=20)
    
    # Step 3: Test the model
    print("\nStep 3: Testing model...")
    test_model(nlp)
    
    # Step 4: Save the model
    print("\nStep 4: Saving model...")
    model_path = save_model(nlp)
    
    print(f"""
Training completed successfully!

Model saved to: {model_path}

To use this model in your app, update app.py:

   models_to_try = [
       "./custom_urgency_model",  # Your custom model
       "en_core_web_trf",         # Fallback
   ]

The model now recognizes these entity types:
   - PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, etc. (standard spaCy)
   - URGENCY (your custom entity)

Urgency keywords detected:
   urgent, asap, immediately, emergency, critical, rush, 
   deadline, priority, soon, quick, fast, now, etc.
""")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
