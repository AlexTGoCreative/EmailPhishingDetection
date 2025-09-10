# Email Phishing Detection Pipeline (ML Core)

A robust, ML-driven phishing detection pipeline that classifies email risks and injects contextual warning banners into messages. The system combines intent detection, tone analysis, grammar checking, and email authentication to provide comprehensive phishing protection.

## 🎯 Overview

This project implements a comprehensive email phishing detection system that:

- **Detects phishing-related intents** (payment requests, credential harvesting)
- **Assesses urgency/tone** relevant to social engineering attacks
- **Evaluates grammar/fluency issues** as weak phishing indicators
- **Combines all signals** into a unified Risk Score
- **Produces contextual banner messages** for user awareness

## 🚀 Features

### Core ML Components
- **Intent Detector**: Multi-label semantic classification using MiniLM-based models
- **Tone/Emotion Analyzer**: Maps tone to phishing-relevant categories (urgency, authority, fear, reward)
- **Grammar/Fluency Checker**: Computes grammar scores and identifies fluency issues
- **Header Analysis**: SPF, DKIM, DMARC validation and received header analysis

### Risk Assessment
- **Unified Risk Scoring**: Combines intent, tone, grammar, and header analysis
- **Configurable Weights**: Tunable parameters for different signal types
- **Banner Selection**: Contextual warning messages based on risk levels

### Input/Output
- **Input**: `.eml` file parsing and normalization
- **Output**: JSON API with risk scores and banner recommendations
- **Banner Injection**: HTML warning blocks for suspicious emails

## 📋 Requirements

### Functional Requirements
- Parse `.eml` files for structured inputs
- Normalize email text (remove signatures, disclaimers, reply chains)
- ML inference stack execution
- Signal fusion into unified Risk Score
- Banner selection and injection
- JSON API output

### Non-Functional Requirements
- **Accuracy**: ≥90% precision/recall on labeled validation dataset
- **Latency**: ≤500ms per email on standard hardware
- **Explainability**: Every prediction outputs contributing tokens/tones
- **Security**: No PII leakage in logs
- **False Positive Rate**: ≤5% on validation dataset

## 🏗️ Architecture

### Pipeline Flow
```
.eml Input → Parsing → Normalization → ML Stack → Risk Scoring → Banner Selection → JSON Output
```

### ML Stack Components
1. **Intent Detector (MiniLM-based)**
   - Multi-label semantic classification
   - Output: `{label: [contributing tokens]}`

2. **Tone/Emotion Analyzer**
   - Maps tone to phishing categories
   - Output: `{tone, confidence}`

3. **Grammar/Fluency Checker**
   - Computes grammar_score (0–1)
   - Returns major grammar/fluency issues

### Risk Score Formula
```
RiskScore = w1 × Intent + w2 × Tone + w3 × Grammar
```
Where `w1`, `w2`, `w3` are tunable weights.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd GrammarAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python setup_models.py
```

### Dependencies
```txt
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
email-validator>=1.2.0
dnspython>=2.2.0
```

## 🚀 Quick Start

### Basic Usage
```python
from phishing_detector import PhishingDetector

# Initialize detector
detector = PhishingDetector()

# Analyze email
result = detector.analyze_email("path/to/email.eml")

# Get risk score and banner
print(f"Risk Score: {result['risk_score']}")
print(f"Banner: {result['banner_message']}")
```

### API Usage
```python
# Batch processing
emails = ["email1.eml", "email2.eml", "email3.eml"]
results = detector.analyze_batch(emails)

# Get detailed analysis
for email_path, analysis in results.items():
    print(f"Email: {email_path}")
    print(f"Intent: {analysis['intent']}")
    print(f"Tone: {analysis['tone']}")
    print(f"Grammar Score: {analysis['grammar_score']}")
    print(f"Risk Score: {analysis['risk_score']}")
    print("---")
```

## 📊 Risk Scoring

### Banner Severity Levels
- **0–29**: Informational/Benign → No banner injected
- **30–69**: Suspicious → Yellow banner
- **70–100**: High-Risk → Red banner

### Example Banner Messages
- `"Suspicious payment request detected. Proceed with caution."`
- `"High-risk credential harvesting attempt identified. Do not click links."`
- `"Urgent tone detected with suspicious intent. Verify sender authenticity."`

## 🔧 Configuration

### Model Weights
```python
# Customize risk scoring weights
config = {
    "intent_weight": 0.5,
    "tone_weight": 0.3,
    "grammar_weight": 0.2
}

detector = PhishingDetector(config=config)
```

### Banner Customization
```python
# Custom banner templates
banner_config = {
    "suspicious_template": "⚠️ Warning: {reason} detected",
    "high_risk_template": "🚨 ALERT: {reason} - Do not proceed",
    "custom_styles": {
        "suspicious": "background-color: #fff3cd; border: 1px solid #ffeaa7;",
        "high_risk": "background-color: #f8d7da; border: 1px solid #f5c6cb;"
    }
}
```

## 📈 Performance

### Benchmarks
- **Processing Time**: <500ms per email
- **Accuracy**: 90%+ precision/recall
- **False Positive Rate**: <5%
- **Memory Usage**: <2GB for standard models

### Optimization Tips
- Use GPU acceleration for faster inference
- Implement caching for repeated analyses
- Batch process multiple emails for efficiency

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_intent_detection.py
python -m pytest tests/test_risk_scoring.py
python -m pytest tests/test_banner_injection.py

# Run with coverage
python -m pytest --cov=phishing_detector tests/
```

### Test Data
```bash
# Download test datasets
python download_test_data.py

# Validate on test set
python validate_models.py --dataset test_emails/
```

## 📚 API Reference

### PhishingDetector Class

#### Methods
- `analyze_email(eml_path)`: Analyze single email
- `analyze_batch(eml_paths)`: Analyze multiple emails
- `get_model_info()`: Get loaded model information
- `update_weights(weights)`: Update risk scoring weights

#### Output Format
```json
{
    "risk_score": 75,
    "intent": {
        "payment_request": 0.8,
        "credential_harvest": 0.3
    },
    "tone": {
        "urgency": 0.9,
        "authority": 0.2
    },
    "grammar_score": 0.6,
    "banner_message": "Suspicious payment request detected. Proceed with caution.",
    "banner_severity": "high_risk",
    "contributing_tokens": ["urgent", "payment", "immediately"],
    "header_analysis": {
        "spf": "pass",
        "dkim": "fail",
        "dmarc": "fail"
    }
}
```

## 🔍 Error Handling

### Common Error Scenarios
- **Corrupted .eml files**: Returns error with clear message
- **Missing headers**: Continues with reduced confidence
- **Model timeouts**: Fails fast with timeout error
- **Banner injection failures**: Returns JSON without altering email

### Error Response Format
```json
{
    "error": "invalid_eml_format",
    "message": "Unable to parse email file",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 phishing_detector/
black phishing_detector/
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [API Documentation](docs/api.md)
- [Model Architecture](docs/architecture.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### Contact
- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)
- Email: support@your-domain.com

## 🗺️ Roadmap

### Phase 1 (Current)
- [x] Core ML pipeline implementation
- [x] Basic risk scoring
- [x] Banner injection
- [x] JSON API output

### Phase 2 (Planned)
- [ ] Real-time mail gateway integration
- [ ] Advanced VIP impersonation detection
- [ ] Multi-language support
- [ ] Continuous model retraining

### Phase 3 (Future)
- [ ] Distributed infrastructure support
- [ ] External threat intelligence integration
- [ ] Advanced behavioral analysis
- [ ] Mobile app integration

## 🙏 Acknowledgments

- HuggingFace for transformer models
- The cybersecurity community for phishing datasets
- Contributors and testers

---

**⚠️ Disclaimer**: This tool is designed to assist in phishing detection but should not be the sole security measure. Always combine with other security practices and human verification.
