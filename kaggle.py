#!/usr/bin/env python3
"""
Script to train a custom spaCy model with URGENCY entity recognition
Uses training_data.py generated from formatted_phishing_results_with_urgency.txt
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from pathlib import Path
import warnings
import os
import importlib.util
import sys

def validate_training_data(training_data):
    """Validate training data format"""
    if not training_data:
        return False, "No training data provided"
    
    valid_count = 0
    for i, (text, entities_dict) in enumerate(training_data):
        if not isinstance(text, str):
            print(f"Warning: Text at index {i} is not a string")
            continue
        if not isinstance(entities_dict, dict):
            print(f"Warning: Entities at index {i} is not a dictionary")
            continue
        if 'entities' not in entities_dict:
            print(f"Warning: No 'entities' key at index {i}")
            continue
        if not isinstance(entities_dict['entities'], list):
            print(f"Warning: Entities at index {i} is not a list")
            continue
        valid_count += 1
    
    if valid_count == 0:
        return False, "No valid training examples found"
    
    print(f"✓ Validated {valid_count}/{len(training_data)} training examples")
    return True, f"Validated {valid_count} examples"

def load_training_data_from_python_file(path="/kaggle/input/python/training_data.py"):
    """Load training data by importing training_data.py as a module"""
    if not os.path.exists(path):
        print(f"✗ Training data file not found at {path}")
        return []
    
    print(f"✓ Loading training data from {path}")
    spec = importlib.util.spec_from_file_location("training_data_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["training_data_module"] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, "training_data"):
        print(f"✗ No 'training_data' variable found in {path}")
        return []
    
    training_data = module.training_data
    is_valid, message = validate_training_data(training_data)
    if not is_valid:
        print(f"✗ Training data validation failed: {message}")
        return []
    
    print(f"✓ Loaded {len(training_data)} training examples")
    print("\nFirst 3 training examples:")
    for i, (text, entities) in enumerate(training_data[:3], 1):
        print(f"{i}. Text: '{text[:50]}...'")
        print(f"   Entities: {entities}")
    
    return training_data

def find_correct_offsets(text, word):
    """Find correct character offsets for a word in text using spaCy tokenization"""
    import spacy
    nlp = spacy.blank("en")
    doc = nlp(text)
    
    for token in doc:
        if token.text.lower() == word.lower():
            return [(token.idx, token.idx + len(token.text), "URGENCY")]
    return []

def augment_training_data(training_data):
    """Augment training data with variations to increase dataset size"""
    augmented = []
    
    # Add original data
    augmented.extend(training_data)
    
    # Create extensive variations of urgency patterns
    urgency_patterns = [
        ("urgent", "URGENCY"),
        ("asap", "URGENCY"), 
        ("immediately", "URGENCY"),
        ("emergency", "URGENCY"),
        ("critical", "URGENCY"),
        ("rush", "URGENCY"),
        ("priority", "URGENCY"),
        ("now", "URGENCY"),
        ("quickly", "URGENCY"),
        ("fast", "URGENCY"),
        ("deadline", "URGENCY"),
        ("expires", "URGENCY"),
        ("expiring", "URGENCY"),
        ("soon", "URGENCY"),
        ("today", "URGENCY"),
        ("tomorrow", "URGENCY"),
        ("24 hours", "URGENCY"),
        ("immediate", "URGENCY"),
        ("action required", "URGENCY"),
        ("security alert", "URGENCY"),
        ("warning", "URGENCY"),
        ("attention", "URGENCY"),
        ("act fast", "URGENCY"),
        ("click here", "URGENCY"),
        ("verify now", "URGENCY"),
        ("reset now", "URGENCY"),
        ("renew now", "URGENCY"),
        ("confirm now", "URGENCY"),
        ("hurry", "URGENCY"),
        ("hurry up", "URGENCY"),
        ("as soon as possible", "URGENCY"),
        ("right away", "URGENCY"),
        ("instantly", "URGENCY"),
        ("promptly", "URGENCY"),
        ("swiftly", "URGENCY"),
        ("rapidly", "URGENCY"),
        ("time-sensitive", "URGENCY"),
        ("time sensitive", "URGENCY"),
        ("overdue", "URGENCY"),
        ("expired", "URGENCY"),
        ("last chance", "URGENCY"),
        ("final notice", "URGENCY"),
        ("act immediately", "URGENCY"),
        ("respond immediately", "URGENCY"),
        ("take action", "URGENCY"),
        ("urgent action", "URGENCY"),
        ("critical issue", "URGENCY"),
        ("emergency situation", "URGENCY"),
        ("rush order", "URGENCY"),
        ("priority task", "URGENCY"),
        ("high priority", "URGENCY"),
        ("top priority", "URGENCY"),
        ("maximum priority", "URGENCY"),
        ("urgent request", "URGENCY"),
        ("immediate response", "URGENCY"),
        ("quick response", "URGENCY"),
        ("fast response", "URGENCY"),
        ("urgent matter", "URGENCY"),
        ("critical matter", "URGENCY"),
        ("emergency matter", "URGENCY"),
        ("urgent issue", "URGENCY"),
        ("critical issue", "URGENCY"),
        ("emergency issue", "URGENCY"),
        ("urgent problem", "URGENCY"),
        ("critical problem", "URGENCY"),
        ("emergency problem", "URGENCY"),
    ]
    
    # Generate extensive urgency examples with multiple sentence templates
    sentence_templates = [
        # Direct urgency
        "This is {pattern}!",
        "Please respond {pattern}.",
        "We need {pattern} action.",
        "Your account {pattern}.",
        "Please {pattern} your account.",
        "Action {pattern} for this issue.",
        "This requires {pattern} attention.",
        "We need {pattern} response.",
        "Please handle this {pattern}.",
        "This is a {pattern} matter.",
        "We have a {pattern} situation.",
        "This is a {pattern} issue.",
        "We need {pattern} resolution.",
        "Please {pattern} this request.",
        "This needs {pattern} action.",
        "We require {pattern} attention.",
        "Please {pattern} immediately.",
        "This is {pattern} important.",
        "We need {pattern} help.",
        "Please {pattern} as soon as possible.",
        
        # Business context
        "Your {pattern} account needs attention.",
        "Please {pattern} your payment.",
        "We need {pattern} verification.",
        "Your {pattern} subscription expires soon.",
        "Please {pattern} your information.",
        "We need {pattern} confirmation.",
        "Your {pattern} request is pending.",
        "Please {pattern} your details.",
        "We need {pattern} update.",
        "Your {pattern} status requires action.",
        
        # Security context
        "Security {pattern} detected on your account.",
        "Please {pattern} your password.",
        "We detected {pattern} activity.",
        "Your account {pattern} security breach.",
        "Please {pattern} your security settings.",
        "We need {pattern} verification for security.",
        "Your {pattern} access has been compromised.",
        "Please {pattern} your login credentials.",
        "We detected {pattern} login attempts.",
        "Your {pattern} account is at risk.",
        
        # Email/Communication context
        "Please {pattern} to this email.",
        "We need {pattern} reply.",
        "Your {pattern} message requires response.",
        "Please {pattern} your communication.",
        "We need {pattern} feedback.",
        "Your {pattern} inquiry needs attention.",
        "Please {pattern} your questions.",
        "We need {pattern} clarification.",
        "Your {pattern} request needs processing.",
        "Please {pattern} your concerns.",
    ]
    
    # Generate examples for each pattern
    for pattern, label in urgency_patterns:
        for template in sentence_templates:
            try:
                sentence = template.format(pattern=pattern)
                # Find the pattern in the sentence
                start = sentence.lower().find(pattern.lower())
                if start != -1:
                    end = start + len(pattern)
                    entities = [(start, end, label)]
                    augmented.append((sentence, {"entities": entities}))
            except:
                # Skip if template doesn't work with this pattern
                continue
    
    # Add specific examples to fix "now" vs DATE conflict with correct offsets
    now_urgency_examples = [
        ("Please respond now!", {"entities": find_correct_offsets("Please respond now!", "now")}),
        ("We need this done now!", {"entities": find_correct_offsets("We need this done now!", "now")}),
        ("Act now before it's too late!", {"entities": find_correct_offsets("Act now before it's too late!", "now")}),
        ("Submit your application now!", {"entities": find_correct_offsets("Submit your application now!", "now")}),
        ("Call me now!", {"entities": find_correct_offsets("Call me now!", "now")}),
        ("Do it now!", {"entities": find_correct_offsets("Do it now!", "now")}),
        ("Start now!", {"entities": find_correct_offsets("Start now!", "now")}),
        ("Finish now!", {"entities": find_correct_offsets("Finish now!", "now")}),
        ("Stop now!", {"entities": find_correct_offsets("Stop now!", "now")}),
        ("Go now!", {"entities": find_correct_offsets("Go now!", "now")}),
        ("Pay now!", {"entities": find_correct_offsets("Pay now!", "now")}),
        ("Login now!", {"entities": find_correct_offsets("Login now!", "now")}),
        ("Register now!", {"entities": find_correct_offsets("Register now!", "now")}),
        ("Confirm now!", {"entities": find_correct_offsets("Confirm now!", "now")}),
        ("Update now!", {"entities": find_correct_offsets("Update now!", "now")}),
        ("Verify now!", {"entities": find_correct_offsets("Verify now!", "now")}),
        ("Complete now!", {"entities": find_correct_offsets("Complete now!", "now")}),
        ("Submit now!", {"entities": find_correct_offsets("Submit now!", "now")}),
        ("Download now!", {"entities": find_correct_offsets("Download now!", "now")}),
        ("Install now!", {"entities": find_correct_offsets("Install now!", "now")}),
    ]
    
    # Add examples where "now" should NOT be URGENCY (context matters)
    now_date_examples = [
        ("I will see you now", {"entities": []}),  # "now" as time reference
        ("Right now I am working", {"entities": []}),  # "now" as current time
        ("From now on we will", {"entities": []}),  # "now" as time reference
        ("Until now everything was fine", {"entities": []}),  # "now" as time reference
        ("Now I understand", {"entities": []}),  # "now" as time reference
        ("Now that you mention it", {"entities": []}),  # "now" as time reference
        ("Now and then I go", {"entities": []}),  # "now" as time reference
        ("Now is the time", {"entities": []}),  # "now" as time reference
        ("Now we can proceed", {"entities": []}),  # "now" as time reference
        ("Now I see the problem", {"entities": []}),  # "now" as time reference
    ]
    
    # Add negative examples (no urgency)
    negative_examples = [
        ("This is a regular meeting", {"entities": []}),
        ("Please review when you have time", {"entities": []}),
        ("The report will be ready next week", {"entities": []}),
        ("Thank you for your cooperation", {"entities": []}),
        ("We appreciate your patience", {"entities": []}),
        ("This is not urgent", {"entities": []}),
        ("Take your time with this", {"entities": []}),
        ("No rush on this task", {"entities": []}),
        ("This can wait", {"entities": []}),
        ("No immediate action needed", {"entities": []}),
        ("This is a normal process", {"entities": []}),
        ("Standard procedure applies", {"entities": []}),
        ("Regular business hours", {"entities": []}),
        ("This is not an emergency", {"entities": []}),
        ("No special action required", {"entities": []}),
    ]
    
    augmented.extend(now_urgency_examples)
    augmented.extend(now_date_examples)
    augmented.extend(negative_examples)
    
    print(f"✓ Augmented training data from {len(training_data)} to {len(augmented)} examples")
    print(f"✓ Generated {len(augmented) - len(training_data)} new examples for training")
    return augmented

def get_fallback_training_data():
    """Fallback training data if Kaggle dataset fails to load"""
    return [
        # Clear URGENCY examples - specific urgency keywords
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
        ("This is a priority task", {"entities": [(10, 18, "URGENCY")]}),
        ("We need this done as soon as possible", {"entities": [(20, 23, "URGENCY")]}),
        ("Please expedite this request", {"entities": [(7, 15, "URGENCY")]}),
        ("This requires immediate attention", {"entities": [(15, 24, "URGENCY")]}),
        ("Time-sensitive matter needs resolution", {"entities": [(0, 13, "URGENCY")]}),
        
        # Clear NON-urgency examples - words that should NOT be tagged as URGENCY
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
    ]

def create_blank_model():
    """Create a blank spaCy model with NER component"""
    # Suppress warnings about missing lookup tables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Create blank model
        nlp = spacy.blank("en")
        
        # Remove ALL components that might cause lookup table issues
        components_to_remove = ["lemmatizer", "attribute_ruler", "tok2vec"]
        for component in components_to_remove:
            if component in nlp.pipe_names:
                nlp.remove_pipe(component)
        
        # Add NER component
        ner = nlp.add_pipe("ner")
        
        # Add the new label
        ner.add_label("URGENCY")
    
    return nlp

def create_minimal_model():
    """Create a minimal spaCy model with just tokenizer and NER"""
    from spacy.lang.en import English
    from spacy.pipeline.ner import NER
    
    # Create minimal English language class
    nlp = English()
    
    # Add only NER component
    ner = nlp.add_pipe("ner")
    ner.add_label("URGENCY")
    
    return nlp

def update_existing_model():
    """Load en_core_web_trf and add URGENCY label safely"""
    model_name = "en_core_web_trf"
    try:
        print(f"Loading {model_name}...")
        nlp = spacy.load(model_name)
        print(f"✓ Loaded {model_name} successfully")
        
        # Remove components that might cause lookup table issues
        components_to_remove = ["lemmatizer", "attribute_ruler"]
        for component in components_to_remove:
            if component in nlp.pipe_names:
                nlp.remove_pipe(component)
                print(f"✓ Removed {component} to avoid lookup table issues")
        
        # Get NER component
        ner = nlp.get_pipe("ner")
        
        # Add new label
        ner.add_label("URGENCY")
        print("✓ Added URGENCY label to NER")
        
        return nlp
    except Exception as e:
        print(f"✗ Could not load {model_name}: {e}")
        print("Falling back to blank model...")
        return create_blank_model()

def train_model(nlp, training_data, iterations=20):
    """Train the NER model - bypasses initialization issues"""
    ner = nlp.get_pipe("ner")
    
    # Convert training data to Example objects with progress tracking
    print(f"Converting {len(training_data)} training examples...")
    examples = []
    for i, (text, ann) in enumerate(training_data):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(training_data)} examples...")
        try:
            examples.append(Example.from_dict(nlp.make_doc(text), ann))
        except Exception as e:
            print(f"  Warning: Skipping example {i}: {e}")
            continue
    
    print(f"Successfully converted {len(examples)} examples")
    
    print(f"\n🚀 Training NER for {iterations} iterations...")
    print(f"📊 Training with {len(examples)} examples")
    print("=" * 60)
    
    for itn in range(iterations):
        # Progress bar
        progress = (itn + 1) / iterations
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        percentage = progress * 100
        
        print(f"\n�� Iteration {itn+1:2d}/{iterations} [{bar}] {percentage:5.1f}%")
        
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=8)
        
        # Process batches with progress tracking
        batch_count = 0
        for batch in batches:
            try:
                nlp.update(batch, drop=0.1, losses=losses)
                batch_count += 1
                
                # Show progress every 50 batches
                if batch_count % 50 == 0:
                    print(f"  Processed {batch_count} batches...")
                    
            except Exception as e:
                print(f"  Error in batch {batch_count}: {e}")
                continue
        
        # Display losses with emojis
        loss_info = []
        if 'ner' in losses:
            loss_info.append(f"NER: {losses['ner']:.2f}")
        if 'transformer' in losses:
            loss_info.append(f"TRF: {losses['transformer']:.2f}")
        if 'tagger' in losses and losses['tagger'] > 0:
            loss_info.append(f"TAG: {losses['tagger']:.2f}")
        if 'parser' in losses and losses['parser'] > 0:
            loss_info.append(f"PAR: {losses['parser']:.2f}")
        
        print(f"   📈 Losses: {' | '.join(loss_info)}")
        
        # Early stopping check
        if 'ner' in losses and losses['ner'] < 10:
            print(f"   ✅ NER loss very low ({losses['ner']:.2f}), stopping early!")
            break
    
    print(f"\n�� Training completed! Processed {len(examples)} examples")
    return nlp

def test_model(nlp):
    """Test the trained model with sample texts"""
    test_texts = [
        # URGENCY examples from training data
        "Warning: Unusual login attempt detected on your Amazon. Verify your identity.",
        "Urgent! Your Google has been compromised. Click here to secure it now!",
        "Immediate action required: Your Spotify subscription is expiring. Renew now.",
        "Act fast! Your HSBC account will be locked if not verified within 24 hours.",
        "Security alert: We have detected suspicious activity on your Outlook. Reset your password now!",
        
        # Simple URGENCY examples
        "This is urgent! Please respond ASAP.",
        "Emergency meeting with John Smith at Microsoft.",
        "Critical deadline tomorrow for the project.",
        "Rush delivery needed now!",
        
        # NON-urgency examples
        "Regular meeting next week in New York.",
        "Also, we need to discuss the budget.",
        "Best regards, Mike Anderson",
        "Google Inc. is a technology company",
    ]
    
    print("\nTesting trained model:")
    print("=" * 50)
    
    urgency_found = 0
    total_urgency_tests = 9  # First 9 tests should have URGENCY
    
    for i, text in enumerate(test_texts):
        doc = nlp(text)
        print(f"\nTest {i+1}: '{text}'")
        print("Entities found:")
        
        has_urgency = False
        if doc.ents:
            for ent in doc.ents:
                if ent.label_ == 'URGENCY':
                    has_urgency = True
                    urgency_found += 1
                print(f"  - '{ent.text}' -> {ent.label_} ({spacy.explain(ent.label_) if ent.label_ != 'URGENCY' else 'Urgency indicator'})")
        else:
            print("  - No entities found")
        
        # Check if this test should have URGENCY
        if i < total_urgency_tests and not has_urgency:
            print("  ❌ MISSING URGENCY - This should have been detected as urgent!")
        elif i >= total_urgency_tests and has_urgency:
            print("  ❌ FALSE POSITIVE - This should NOT be urgent!")
        elif i < total_urgency_tests and has_urgency:
            print("  ✅ CORRECT - URGENCY detected!")
        else:
            print("  ✅ CORRECT - No URGENCY (as expected)")
    
    print(f"\n📊 URGENCY Detection Summary:")
    print(f"   Found: {urgency_found}/{total_urgency_tests} expected URGENCY entities")
    print(f"   Accuracy: {(urgency_found/total_urgency_tests)*100:.1f}%")
    
    if urgency_found == 0:
        print("\n⚠️  WARNING: Model is not detecting ANY URGENCY entities!")
        print("   This suggests the model needs more training or different approach.")

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
    
    # Step 1: Load training data from generated Python file
    print("\nStep 1: Loading training data from training_data.py...")
    training_data = load_training_data_from_python_file("/kaggle/input/python4/training_data.py")
    
    if not training_data:
        print("✗ No training data available. Using fallback data...")
        training_data = get_fallback_training_data()
    
    # Step 1.5: Augment training data to increase dataset size
    print("\nStep 1.5: Augmenting training data...")
    training_data = augment_training_data(training_data)
    
    # Step 2: Load or create model
    print("\nStep 2: Loading base model...")
    try:
        nlp = update_existing_model()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Falling back to blank model...")
        nlp = create_blank_model()
    
    # Step 3: Train the model
    print("\nStep 3: Training model...")
    nlp = train_model(nlp, training_data, iterations=5)
    
    # Step 4: Test the model
    print("\nStep 4: Testing model...")
    test_model(nlp)
    
    # Step 5: Save the model
    print("\nStep 5: Saving model...")
    model_path = save_model(nlp)
    
    print(f"\n✓ Training completed! Model saved to {model_path}")

if __name__ == "__main__":
    random.seed(42)
    main()
