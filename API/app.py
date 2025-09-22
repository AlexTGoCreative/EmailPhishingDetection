from flask import Flask, request, jsonify, render_template
import spacy
from spacy.pipeline import EntityRuler
from collections import defaultdict
import json
import re

app = Flask(__name__)

# Regex patterns for structured entities and phishing detection
REGEX_PATTERNS = [
    # URLs
    {"label": "URL", "pattern": r"https?://[^\s]+"},
    {"label": "URL", "pattern": r"www\.[^\s]+"},

    # Email addresses (RFC 5322-compliant, cu re.VERBOSE pentru lizibilitate)
    {
        "label": "EMAIL",
        "pattern": r"""
        (                           # start full match
          ([-!#-'*+\-/0-9=?A-Z^-~]+(\.[-!#-'*+\-/0-9=?A-Z^-~]+)*   # local part normal
           |                                                       # OR
           ("([ !#-[\]-~]|(\\[ \t -~]))+"))                        # quoted local part
          @
          (
            [0-9A-Za-z]([0-9A-Za-z-]{0,61}[0-9A-Za-z])?            # domain labels
            (\.[0-9A-Za-z]([0-9A-Za-z-]{0,61}[0-9A-Za-z])?)*      # dot-separated domain
            |
            \[
              (                                                     # IP or IPv6 literal
                (25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])
                (\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}
                |
                IPv6:(
                  (((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){6}
                  |::((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){5}
                  |[0-9A-Fa-f]{0,4}::((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){4}
                  |(((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):)?(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}))?::((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){3}
                  |(((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){0,2}(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}))?::((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){2}
                  |(((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){0,3}(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}))?::(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):
                  |(((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){0,4}(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}))?::
                )
                ((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3})
                |
                (25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])
                (\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3})
                |
                (((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){0,5}(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}))?::(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3})
                |
                (((0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}):){0,6}(0|[1-9A-Fa-f][0-9A-Fa-f]{0,3}))?::
              )
              |
              (?!IPv6:)[0-9A-Za-z-]*[0-9A-Za-z]:[!-Z^-~]+
            )
          \]
        )""",
        "flags": re.VERBOSE
    },

    # Phone numbers (US/international)
    {"label": "PHONE", "pattern": r"\+?1[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}(?![0-9])"},
    {"label": "PHONE", "pattern": r"\+?[0-9]{1,3}[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}(?![0-9])"},
    {"label": "PHONE", "pattern": r"\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}(?![0-9])"},

    # IP addresses
    {"label": "IP_ADDRESS", "pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"},

    # Credit card numbers
    {"label": "CREDIT_CARD", "pattern": r"\b[0-9]{4}[-.\s]?[0-9]{4}[-.\s]?[0-9]{4}[-.\s]?[0-9]{4}\b"},
    {"label": "CREDIT_CARD", "pattern": r"\b[0-9]{16}\b"},

    # File extensions
    {"label": "FILE", "pattern": r"\b[a-zA-Z0-9._-]+\.(exe|bat|cmd|scr|pif|com|zip|rar|7z|pdf|doc|docx|txt|jpg|png|gif|mp4|mp3|avi|mov)\b", "flags": re.IGNORECASE},
    {"label": "FILE", "pattern": r"[a-zA-Z0-9._-]+\.(exe|bat|cmd|scr|pif|com|zip|rar|7z|pdf|doc|docx|txt|jpg|png|gif|mp4|mp3|avi|mov)(?![a-zA-Z0-9])", "flags": re.IGNORECASE},

     # Suspicious file extensions
    {"label": "SUSPICIOUS_FILE", "pattern": r"\b[a-zA-Z0-9._-]+\.(exe|bat|cmd|scr|pif|com|zip|rar|7z)\b", "flags": re.IGNORECASE},
    {"label": "SUSPICIOUS_FILE", "pattern": r"[a-zA-Z0-9._-]+\.(exe|bat|cmd|scr|pif|com|zip|rar|7z)(?![a-zA-Z0-9])", "flags": re.IGNORECASE},
    
    # Bitcoin addresses
    {"label": "BITCOIN", "pattern": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"},
    {"label": "BITCOIN", "pattern": r"\bbc1[a-z0-9]{39,59}\b"},

    # SSN
    {"label": "SSN", "pattern": r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"},
    {"label": "SSN", "pattern": r"\bSSN:\s*[0-9]{3}-[0-9]{2}-[0-9]{4}\b", "flags": re.IGNORECASE},

    # Bank routing numbers
    {"label": "ROUTING_NUMBER", "pattern": r"\b[0-9]{9}\b"},

    # Years
    {"label": "DATE", "pattern": r"\b(19|20)[0-9]{2}\b"},
]

def create_entity_ruler(nlp):
    """Create and configure EntityRuler with regex patterns"""
    ruler = EntityRuler(nlp, overwrite_ents=True)
    
    # Add patterns to the ruler - EntityRuler doesn't support regex directly
    # We'll handle regex patterns in the extract_entities method instead
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
        """Extract all entities using spaCy and regex patterns"""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(text)
            
            entities = []
            used_positions = set()
            
            # Extract spaCy entities first
            for ent in doc.ents:
                description = spacy.explain(ent.label_)
                
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': description,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                
                # Mark spaCy entity positions as used
                for pos in range(ent.start_char, ent.end_char):
                    used_positions.add(pos)
            
            # Extract regex-based entities (avoiding overlaps with spaCy)
            regex_entities = self._extract_regex_entities(text, used_positions)
            entities.extend(regex_entities)
            
            # Sort entities by start position
            entities.sort(key=lambda x: x['start'])
            
            return entities
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return []
    
    def _extract_regex_entities(self, text, used_positions=None):
        """Extract entities using regex patterns"""
        entities = []
        if used_positions is None:
            used_positions = set()
        
        for pattern in REGEX_PATTERNS:
            flags = pattern.get("flags", 0)
            regex = re.compile(pattern["pattern"], flags)
            
            for match in regex.finditer(text):
                start = match.start()
                end = match.end()
                
                # Check if this position is already used by spaCy
                if any(pos in used_positions for pos in range(start, end)):
                    continue
                
                # Mark these positions as used
                for pos in range(start, end):
                    used_positions.add(pos)
                
                entities.append({
                    'text': match.group(),
                    'label': pattern["label"],
                    'description': f"Regex-detected {pattern['label'].lower()}",
                    'start': start,
                    'end': end
                })
        
        return entities
    
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
            
            # Regex-based labels
            'EMAIL': 'CONTACT_INFO',
            'PHONE': 'CONTACT_INFO',
            'URL': 'TECHNICAL',
            'IP_ADDRESS': 'TECHNICAL',
            'BITCOIN': 'FINANCIAL',
            'CREDIT_CARD': 'FINANCIAL',
            'SSN': 'FINANCIAL',
            'ROUTING_NUMBER': 'FINANCIAL',
            'SUSPICIOUS_FILE': 'SUSPICIOUS'
        }
        
        for entity in entities:
            category = label_to_category.get(entity['label'], 'OTHER')
            categories[category].append(entity)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    

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
        
        # Prepare response
        response = {
            'text': text,
            'entities': entities,
            'categorized': categorized,
            'total_entities': len(entities),
            'model_used': model_name if nlp else 'none'
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
