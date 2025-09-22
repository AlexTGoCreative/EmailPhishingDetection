from flask import Flask, request, jsonify, render_template
import spacy
from spacy.pipeline import EntityRuler
from collections import defaultdict
import json
import re

app = Flask(__name__)

# Regex patterns for structured entities and phishing detection
REGEX_PATTERNS = [
    # URLs and domains
    {"label": "URL", "pattern": r"https?://[^\s]+"},
    {"label": "URL", "pattern": r"www\.[^\s]+"},
    {"label": "DOMAIN", "pattern": r"[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}"},
    
    # Email addresses
    {"label": "EMAIL", "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"},
    
    # Phone numbers (various formats)
    {"label": "PHONE", "pattern": r"\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"},
    {"label": "PHONE", "pattern": r"\+?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}"},
    
    # IP addresses
    {"label": "IP_ADDRESS", "pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"},
    
    # Credit card numbers (basic pattern)
    {"label": "CREDIT_CARD", "pattern": r"\b[0-9]{4}[-.\s]?[0-9]{4}[-.\s]?[0-9]{4}[-.\s]?[0-9]{4}\b"},
    
    # Suspicious domains (common phishing patterns)
    {"label": "SUSPICIOUS_DOMAIN", "pattern": r"[a-zA-Z0-9.-]*\.tk\b"},
    {"label": "SUSPICIOUS_DOMAIN", "pattern": r"[a-zA-Z0-9.-]*\.ml\b"},
    {"label": "SUSPICIOUS_DOMAIN", "pattern": r"[a-zA-Z0-9.-]*\.ga\b"},
    {"label": "SUSPICIOUS_DOMAIN", "pattern": r"[a-zA-Z0-9.-]*\.cf\b"},
    
    # Urgency keywords
    {"label": "URGENCY", "pattern": r"\b(urgent|asap|immediately|expires?|deadline|limited time|act now|don't wait)\b", "flags": re.IGNORECASE},
    {"label": "URGENCY", "pattern": r"\b(verify|confirm|update|suspended|locked|expired|security|breach)\b", "flags": re.IGNORECASE},
    
    # Financial urgency
    {"label": "FINANCIAL_URGENCY", "pattern": r"\b(account|payment|billing|invoice|overdue|penalty|fine|charge)\b", "flags": re.IGNORECASE},
    
    # Suspicious file extensions
    {"label": "SUSPICIOUS_FILE", "pattern": r"\b[a-zA-Z0-9._-]+\.(exe|bat|cmd|scr|pif|com|zip|rar|7z)\b", "flags": re.IGNORECASE},
    
    # Bitcoin addresses
    {"label": "BITCOIN", "pattern": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"},
    {"label": "BITCOIN", "pattern": r"\bbc1[a-z0-9]{39,59}\b"},
    
    # Social Security Numbers (US format)
    {"label": "SSN", "pattern": r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"},
    
    # Bank routing numbers
    {"label": "ROUTING_NUMBER", "pattern": r"\b[0-9]{9}\b"},
]

def create_entity_ruler(nlp):
    """Create and configure EntityRuler with regex patterns"""
    ruler = EntityRuler(nlp, overwrite_ents=True)
    
    # Add patterns to the ruler
    for pattern in REGEX_PATTERNS:
        flags = pattern.get("flags", 0)
        ruler.add_patterns([{
            "label": pattern["label"],
            "pattern": [{"TEXT": {"REGEX": pattern["pattern"]}}]
        }])
    
    return ruler

# Load spaCy model once when the app starts
def load_best_available_model():
    """Load the en_core_web_trf model and add EntityRuler"""
    model_name = "en_core_web_trf"
    
    try:
        nlp = spacy.load(model_name)
        print(f"✓ Loaded {model_name} successfully")
        
        # Add EntityRuler to the pipeline
        ruler = create_entity_ruler(nlp)
        nlp.add_pipe("entity_ruler", before="ner")
        print("✓ Added EntityRuler with regex patterns")
        
        return nlp, model_name
    except OSError:
        print(f"✗ {model_name} not available")
        print("Please install the model with:")
        print("python -m spacy download en_core_web_trf")
        raise Exception("en_core_web_trf model not available")

# Initialize the NLP model
try:
    nlp, model_name = load_best_available_model()
    print(f"Using model: {model_name}")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None

class EntityExtractor:
    """Entity extraction class for the Flask app"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
    
    def extract_entities(self, text):
        """Extract all entities using spaCy"""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(text)
            
            entities = []
            for ent in doc.ents:
                description = spacy.explain(ent.label_)
                
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': description,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return []
    
    def categorize_entities(self, entities):
        """Group entities by category"""
        categories = {
            'PEOPLE': [],
            'ORGANIZATIONS': [],
            'LOCATIONS': [],
            'DATES': [],
            'TIMES': [],
            'MONEY': [],
            'QUANTITIES': [],
            'URGENCY': [],
            'CONTACT_INFO': [],
            'TECHNICAL': [],
            'FINANCIAL': [],
            'SUSPICIOUS': [],
            'OTHER': []
        }
        
        label_to_category = {
            # spaCy NER labels
            'PERSON': 'PEOPLE',
            'ORG': 'ORGANIZATIONS',
            'GPE': 'LOCATIONS',  # Countries, cities, states
            'LOC': 'LOCATIONS',  # Mountains, bodies of water
            'DATE': 'DATES',
            'TIME': 'TIMES',
            'MONEY': 'MONEY',
            'QUANTITY': 'QUANTITIES',
            'CARDINAL': 'QUANTITIES',
            'ORDINAL': 'QUANTITIES',
            'PERCENT': 'QUANTITIES',
            'URGENCY': 'URGENCY',
            
            # Regex-based labels
            'EMAIL': 'CONTACT_INFO',
            'PHONE': 'CONTACT_INFO',
            'URL': 'TECHNICAL',
            'DOMAIN': 'TECHNICAL',
            'IP_ADDRESS': 'TECHNICAL',
            'BITCOIN': 'FINANCIAL',
            'CREDIT_CARD': 'FINANCIAL',
            'SSN': 'FINANCIAL',
            'ROUTING_NUMBER': 'FINANCIAL',
            'SUSPICIOUS_DOMAIN': 'SUSPICIOUS',
            'SUSPICIOUS_FILE': 'SUSPICIOUS',
            'FINANCIAL_URGENCY': 'URGENCY'
        }
        
        for entity in entities:
            category = label_to_category.get(entity['label'], 'OTHER')
            categories[category].append(entity)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def analyze_phishing_indicators(self, entities, text):
        """Analyze entities for phishing indicators"""
        phishing_score = 0
        indicators = []
        
        # Count suspicious entities
        suspicious_count = len([e for e in entities if e['label'] in ['SUSPICIOUS_DOMAIN', 'SUSPICIOUS_FILE']])
        urgency_count = len([e for e in entities if e['label'] in ['URGENCY', 'FINANCIAL_URGENCY']])
        financial_count = len([e for e in entities if e['label'] in ['CREDIT_CARD', 'SSN', 'ROUTING_NUMBER', 'BITCOIN']])
        
        # Calculate phishing score
        if suspicious_count > 0:
            phishing_score += suspicious_count * 30
            indicators.append(f"Found {suspicious_count} suspicious domain(s) or file(s)")
        
        if urgency_count > 2:
            phishing_score += urgency_count * 10
            indicators.append(f"High urgency language detected ({urgency_count} instances)")
        
        if financial_count > 0:
            phishing_score += financial_count * 20
            indicators.append(f"Financial information detected ({financial_count} instances)")
        
        # Check for common phishing patterns in text
        text_lower = text.lower()
        if any(word in text_lower for word in ['click here', 'verify now', 'update immediately', 'account suspended']):
            phishing_score += 15
            indicators.append("Common phishing phrases detected")
        
        if any(word in text_lower for word in ['congratulations', 'winner', 'free', 'prize']):
            phishing_score += 10
            indicators.append("Suspicious promotional language detected")
        
        # Determine risk level
        if phishing_score >= 50:
            risk_level = "HIGH"
        elif phishing_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'phishing_score': phishing_score,
            'risk_level': risk_level,
            'indicators': indicators,
            'suspicious_entities': suspicious_count,
            'urgency_entities': urgency_count,
            'financial_entities': financial_count
        }

# Initialize extractor
extractor = EntityExtractor(nlp) if nlp else None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """API endpoint to analyze text and extract entities"""
    try:
        if not extractor:
            return jsonify({'error': 'NLP model not loaded'}), 500
            
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Extract entities
        entities = extractor.extract_entities(text)
        categorized = extractor.categorize_entities(entities)
        phishing_analysis = extractor.analyze_phishing_indicators(entities, text)
        
        # Prepare response
        response = {
            'text': text,
            'entities': entities,
            'categorized': categorized,
            'total_entities': len(entities),
            'model_used': model_name if nlp else 'none',
            'phishing_analysis': phishing_analysis
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze_text: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': nlp is not None,
        'model_name': model_name if nlp else None
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if not nlp:
        print("⚠️  Warning: No spaCy model loaded. Please install with:")
        print("   python -m spacy download en_core_web_trf")
        print("⚠️  The app will start but won't work properly.")
    else:
        print("✅ Entity extraction service ready!")
        print(f"✅ Using model: {model_name}")
        print("✅ Available entity types:", sorted(nlp.get_pipe("ner").labels) if "ner" in nlp.pipe_names else "None")
        print("✅ Regex patterns loaded for structured entities and phishing detection")
        print("✅ Phishing analysis enabled with risk scoring")
    
    print("\n🚀 Starting Flask server...")
    print("🌐 Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
